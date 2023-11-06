using Base.Threads: nthreads, @threads, @spawn
using Base.Iterators: partition # Multithreading
using Random        # Mainly just for randperm()
using StatsBase     # Just for inverse_rle
using NPZ   # Read .npy files
using JSON  # Read json formatted files, duh
using JLD
using DelimitedFiles
using Mmap
using BSplineKit
using DataFrames
using DataFramesMeta
using PooledArrays
using Pipe
using GLMakie
using AlgebraOfGraphics
using CausalityTools
using BenchmarkTools

moths = ["2023-05-20", "2023-05-25"]


vnc_dir = "/Volumes/PikesPeak/VNCMP"
motor_program_dir = "/Volumes/PikesPeak/VNCMP/MP_data/good_data"
analysis_dir = @__DIR__

# Constants everything should know (effectively global)
# NOTE: I suspect LAX and LBA got flipped in split_motor_program.py, so flipping them back here. But true AMPS order is lax, lba
muscle_order = [
    "lba", "lax", "lsa", "ldvm", "ldlm", 
    "rdlm", "rdvm", "rsa", "rba", "rax"]
phase_wrap_thresholds = Dict("ax"=>0.32, "ba"=>0.6, "sa"=>0.4, "dvm"=>3.0, "dlm"=>3.0)


include("IntanReader.jl")
include("functions.jl")
include("precision_functions.jl")
include("match_data.jl")

##


df = vcat(
    read_and_match_moth_data(
        "/Volumes/PikesPeak/VNCMP/2023-05-25", 
        "/Volumes/PikesPeak/VNCMP/MP_data/good_data/2023-05-25_12-24-05"),
    read_and_match_moth_data(
        "/Volumes/PikesPeak/VNCMP/2023-05-20", 
        "/Volumes/PikesPeak/VNCMP/MP_data/good_data/2023-05-20_15-36-35")
)


# Post-processing
df = @pipe df |> 
    @transform(_, :wbfreq = 1 ./ :wblen) |> 
    # Clean out wingbeats below a frequency threshold
    @subset(_, :wbfreq .> 10) |> 
    # rdlm barely has any data so go ahead and remove
    @subset(_, :unit .!= "rdlm") |> 
    # Remove neural units that are noise
    @subset(_, :quality .!= "noise") |> 
    # Final wingbeats have neural spikes that drag on for many seconds
    # May be useful later, but for now removing any neural spikes past when wingbeat should end
    @subset(_, :time .<= :wblen) |> 
    # Make phase column
    @transform(_, :phase = :time ./ :wblen) |> 
    # Unwrap muscle spikes
    groupby(_, [:moth, :poke]) |> 
    transform!(_, [:time, :phase, :wb, :wblen, :unit, :ismuscle] => unwrap_spikes_to_prev => [:time, :phase, :wb, :wblen]) |> 
    @subset(_, (!).(isnan.(:time)))

## Settings for all plots

dark_or_light = "light"
if dark_or_light .== "light"
    set_theme!()
    figsdir = joinpath(analysis_dir, "figs_light")
else
    set_theme!(theme_dark())
    figsdir = joinpath(analysis_dir, "figs_dark")
end
update_theme!(fontsize=30)


## Histograms of muscle spike phase

@pipe df |> 
    @subset(_, :ismuscle) |> 
    @subset(_, :phase .!= 0.0) |> 
    (
    AlgebraOfGraphics.data(_) *
    mapping(:phase, color=:moth, row=:unit) *
    histogram(bins=100, normalization=:pdf, datalimits=extrema) *
    visual(alpha=0.6)
    ) |> 
    draw(_, 
        figure=(resolution=(1000, 1800),),
        facet=(; linkxaxes=:colwise, linkyaxes=:none),
        axis=(; limits=(nothing, nothing))) #|> 
    # save(joinpath(figsdir, "muscle_phase_hist_with_1centered_unwrap.png"), _)
current_figure()

## Dimensionality checking. 
# Max number of spikes per wingbeat combinatorially between muscles and neurons?
max_n_spike_per_wb = @pipe df |> 
    groupby(_, [:moth, :poke, :unit, :wb]) |> 
    combine(_, nrow) |> 
    groupby(_, [:moth, :poke, :unit]) |> 
    combine(_, :nrow => maximum)

# How many rows (wingstrokes) vs columns (max num spikes) for each combination of (neuron, muscle)
nwb_combos = Dict{Tuple, Int}()
for gdf in groupby(df, [:moth, :poke, :wb])
    muscles = unique(gdf[gdf.ismuscle, :unit])
    neurons = unique(gdf[(!).(gdf.ismuscle), :unit])
    for combos in Base.product(muscles, neurons)
        key = (first(gdf.moth), first(gdf.poke), combos[1], combos[2])
        if combos[1] == combos[2]
            continue
        elseif !haskey(nwb_combos, key)
            nwb_combos[key] = 1
        else
            nwb_combos[key] += 1
        end
    end
end

# For each combination get overall dimensionality
num_wingbeats = Int[]
num_dimensions = Int[]
num_dim_firstmuscle = Int[]
for key in keys(nwb_combos)
    append!(num_wingbeats, nwb_combos[key])
    dim1 = @subset(max_n_spike_per_wb, (:moth .== key[1]) .&& (:poke .== key[2]) .&& (:unit .== key[3])).nrow_maximum[1]
    dim2 = @subset(max_n_spike_per_wb, (:moth .== key[1]) .&& (:poke .== key[2]) .&& (:unit .== key[4])).nrow_maximum[1]
    append!(num_dimensions, dim1 + dim2)
    append!(num_dim_firstmuscle, 1 + dim2)
end

f = Figure(resolution=(1200, 900))
ax = Axis(f[1,1],
    xlabel="Number of wingbeats for (muscle, neuron) pairs",
    ylabel="Overall dimensionality (# of columns in input + output)")
scatter!(ax, num_wingbeats, num_dimensions)
save(joinpath(figsdir, "dimensionality_vs_nwingbeats.png"), f)
current_figure()

f = Figure(resolution=(1200, 900))
ax = Axis(f[1,1],
    xlabel="Number of wingbeats for (muscle, neuron) pairs",
    ylabel="Overall dimensionality (# of columns in input + output)")
scatter!(ax, num_wingbeats, num_dimensions, label="Original")
scatter!(ax, num_wingbeats, num_dim_firstmuscle, label="Muscles only first spike")
axislegend()
save(joinpath(figsdir, "dimensionality_vs_nwingbeats_muscles_only_one.png"), f)
current_figure()


##--- Calculate and plot precision for each (muscle, neuron) combination
# Create list of all muscle, unit combinations for each moth
combinations = Dict{String, Set}(moth => Set([]) for moth in unique(df.moth))
for gdf in groupby(df, [:moth, :poke, :wb])
    muscles = unique(gdf[gdf.ismuscle, :unit])
    neurons = unique(gdf[(!).(gdf.ismuscle), :unit])
    for combos in Base.product(muscles, neurons)
        if combos[1] == combos[2]
            continue
        elseif !(combos in combinations[gdf.moth[1]])
            push!(combinations[gdf.moth[1]], combos)
        end
    end
end
# group by moth, poke, loop over (muscle, neuron) combinations, calculate precision and make a plot
precFirst = Dict{Tuple{String, String, String}, Float64}()
dt = DataFrame()
for gdf in groupby(df, :moth)
    thismoth = gdf.moth[1]
    for combo in combinations[thismoth]
        println("Moth $thismoth, Muscle: $(combo[1]), Neuron: $(combo[2])")
        X, Y = XY_array_from_dataframe((combo[1], combo[2]), gdf.unit, gdf.time .* 1000, gdf.wb)
        # # Figure out combinatorics, how many wingbeats match each combination
        # nspikeX, nspikeY = vec(sum((!).(isnan.(X)), dims=2)), vec(sum((!).(isnan.(Y)), dims=2))
        # nspike_combinations = unique(eachrow(hcat(nspikeX, nspikeY)))
        # nwb = [sum((nspikeX .== comb[1]) .&& (nspikeY .== comb[2])) for comb in nspike_combinations]
        # dt = vcat(dt, DataFrame(
        #     :moth => thismoth,
        #     :muscle => combo[1],
        #     :neuron => combo[2], 
        #     :nspike_muscle => [x[1] for x in nspike_combinations],
        #     :nspike_neuron => [x[2] for x in nspike_combinations],
        #     :nwb => nwb
        # ))
        # if size(X,1) <= (size(X,2) + size(Y,2))
        if size(X,1) <= (1 + size(Y,2))
            precFirst[(thismoth, combo[1], combo[2])] = 0.0
            continue
        end
        f = Figure()
        ax = Axis(f[1,1], xscale=log10, xlabel="Added noise amplitude (ms)", ylabel="MI (bits)")
        precision_val = precision(X[:,1], Y;
            noise=exp10.(range(log10(0.05), stop=log10(100), length=200)),
            repeats=100,
            do_plot=true,
            ax=ax
        )
        text!(ax, 0.2, 0.2, text="# wingbeats = $(size(X,1))", fontsize=24, space=:relative)
        text!(ax, 0.2, 0.1, text="dimensionality = $(size(X,2) + size(Y,2))", fontsize=24, space=:relative)
        precFirst[(thismoth, combo[1], combo[2])] = precision_val
        ax.title = "Moth $thismoth, Muscle: $(combo[1]), Neuron: $(combo[2])"
        save(joinpath(figsdir, "precision_curves_only_first_muscle_spike", combo[1], join((thismoth, combo[1], combo[2]), "_") * ".png"), f)
    end
end
# TODO: Better system that saves parameters I ran this under (like number of repeats)
save(joinpath(analysis_dir, "precision_rough_first_pass_muscle_first_spike.jld"), "GOV", precFirst)

##
f = Figure()
ax = Axis(f[1,1], xlabel="Precision (ms)")
vals = collect(values(prec))
hist!(ax, vals[(!).(isnan.(vals))]; bins=100, normalization=:pdf)
save(joinpath(figsdir, "precision_distribution.png"), f)
current_figure()

##

f = Figure()
ax = Axis(f[1,1])
scatter!(ax, dt.nspike_muscle, dt.nwb)
# ax.xlabel = "Muscle N spikes"
# ax.ylabel = "Neuron N spikes"
current_figure()
##
@pipe dt |> 
    @transform(_, :nspike_muscle = :nspike_muscle .+ (0.5 .* rand(nrow(dt)) .- 0.25)) |> 
    (AlgebraOfGraphics.data(_) * 
    mapping(:nspike_muscle => "# of muscle spikes", :nwb => "# of wingbeats") * 
    (histogram(bins=([x-0.5 for x in 1:9], 40)) +
    visual(color=:black, markersize=7, alpha=0.2))) |> 
    draw(_, 
        axis=(; xticks=(1:8, [string(x) for x in 1:8]))) |> 
    save(joinpath(figsdir, "Nmusclespikes_vs_Nwingbeats.png"), _)
current_figure()
@pipe dt |> 
    @transform(_, :nspike_neuron = :nspike_neuron .+ (0.5 .* rand(nrow(dt)) .- 0.25)) |> 
    (AlgebraOfGraphics.data(_) * 
    mapping(:nspike_neuron => "# of neuron spikes", :nwb => "# of wingbeats") * 
    (histogram(bins=([x-0.5 for x in 1:34], 40)) +
    visual(color=:black, markersize=7, alpha=0.2))) |> 
    draw(_,
        figure=(resolution=(1700, 800),),
        axis=(; xticks=(1:33, [string(x) for x in 1:33]))) |> 
    save(joinpath(figsdir, "Nneuronspikes_vs_Nwingbeats.png"), _)
current_figure()

## Test area
gdf = first(groupby(df, :moth))
combo = first(combinations[gdf.moth[1]])
X, Y = XY_array_from_dataframe((combo[1], combo[2]), gdf.unit, gdf.time, gdf.wb)

nspikeX, nspikeY = vec(sum((!).(isnan.(X)), dims=2)), vec(sum((!).(isnan.(Y)), dims=2))
nspike_combinations = unique(eachrow(hcat(nspikeX, nspikeY)))
# nwb = [sum((nspikeX .== comb[1]) .&& (nspikeY .== comb[2])) for comb in nspike_combinations]
# nspikeX, nspikeY = vec(sum((!).(isnan.(x)), dims=2)), vec(sum((!).(isnan.(y)), dims=2))
nspike_combinations = unique(eachrow(hcat(nspikeX, nspikeY)))
probs = Dict(combo => sum((nspikeX .== combo[1]) .&& (nspikeY .== combo[2])) for combo in nspike_combinations)

## Plot of precision curves, facet per muscle


## Plot all units against time for a couple wingstrokes

dt = @subset(df, (:moth .== "2023-05-25") .&& (:wb .>= 1000) .&& (:wb .<= 1004))

f = Figure(resolution=(1800, 1000))
ax = Axis(f[1,1], xlabel="Time (s)")
vlines!(dt[dt.unit .== "ldlm" .&& dt.phase .== 1.0, :abstime];
    color=:grey)
increment = 1 / (length(unique(dt.unit)) + 2)
value_to_index = Dict(key => i for (i, key) in enumerate(unique(dt.wb)))
for (i, (key, gdf)) in enumerate(pairs(groupby(dt, :unit)))
    wbvec = [value_to_index[value] for value in gdf.wb]
    vlines!(ax, gdf.abstime;
        ymin=i * increment - increment * 0.4,
        ymax=i * increment + increment * 0.4,
        linewidth = gdf.ismuscle[1] ? 4 : 1.5,
        # linestyle = gdf.ismuscle[1] ? :dashdot : :solid,
        label=key[1],
        color=wbvec,
        colormap=:tab10,
        colorrange=(1,10))
end
hideydecorations!(ax)
save(joinpath(figsdir, "example_spikes_across_several_wb_1centered.png"), f)
current_figure()

## group by unit, look at ISI to catch overlapping spikes

for (key, gdf) in pairs(groupby(@subset(df, (!).(:ismuscle)), [:moth, :poke, :unit]))
    diftime = diff(gdf.abstime)
    diftime = diftime[diftime .< 0.2]
    if length(diftime) < 1
        continue
    end
    f = Figure()
    ax = Axis(f[1,1])
    hist!(ax, diftime; bins=100, normalization=:pdf)
    save(joinpath(figsdir, "ISI", key[1] * "_" * string(key[3]) * ".png"), f)
end

## Wingbeat frequency plot

@pipe df |> 
    (
    AlgebraOfGraphics.data(_) *
    mapping(:wbfreq => "Wingbeat Frequency (Hz)", color=:moth => "Moth") * 
    histogram(bins=100, normalization=:pdf, datalimits=extrema) *
    visual(alpha=0.5)
    ) |> 
    draw(_,
        figure=(resolution=(1600, 800),)) |> 
    save(joinpath(figsdir, "wingbeat_frequency.png"), _)
current_figure()

##

bob = @pipe df |> 
    groupby(_, [:moth, :poke, :unit]) |> 
    combine(_, nrow)

jim = @pipe df |> 
    @subset(_, (!).(:ismuscle)) |> 
    groupby(_, [:moth, :poke, :unit, :quality]) |> 
    combine(_, nrow)
