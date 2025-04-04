using Base.Threads: nthreads, @threads, @spawn
using Base.Iterators: partition # Multithreading
using Random        # Mainly just for randperm()
using StatsBase     # For inverse_rle, countmap
using NPZ   # Read .npy files
using JSON  # Read json formatted files, duh
# using JLD   # Read and write .jld files for saving dicts
using CSV   # Read and write .csv files, mostly for saving dataframes
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
# # Special process for transfer entropy module
# include("CoTETE.jl/src/CoTETE.jl")
# import .CoTETE

moths = ["2023-05-20", "2023-05-25"]


# vnc_dir = "/Volumes/PikesPeak/VNCMP"
# motor_program_dir = "/Volumes/PikesPeak/VNCMP/MP_data/good_data"
vnc_dir = "/Users/leo/Desktop/ResearchPhD/VNCMP/localdata/"
motor_program_dir = "/Users/leo/Desktop/ResearchPhD/VNCMP/localdata/motor_program"
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

# df = vcat(
#     read_and_match_moth_data(
#         "/Volumes/PikesPeak/VNCMP/2023-05-25", 
#         "/Volumes/PikesPeak/VNCMP/MP_data/good_data/2023-05-25_12-24-05"),
#     read_and_match_moth_data(
#         "/Volumes/PikesPeak/VNCMP/2023-05-20", 
#         "/Volumes/PikesPeak/VNCMP/MP_data/good_data/2023-05-20_15-36-35")
# )
df = read_and_match_moth_data(
    "/Users/leo/Desktop/ResearchPhD/VNCMP/localdata/2023-05-25",
    "/Users/leo/Desktop/ResearchPhD/VNCMP/localdata/motor_program/2023-05-25_12-24-05")

##
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


## Try out transfer entropy

bob = @subset(df, :moth .== "2023-05-25" .&& :poke .== 1)
target = bob[bob.unit .== "ldlm", :abstime]
source = bob[bob.unit .== "18", :abstime]
sort!(target)
sort!(source)
target, source = convert.(Float64, target), convert.(Float64, source)

target = target[target .> source[1]]
# target = target[vcat(diff(target), 0) .> 0.04]

f = Figure()
ax = Axis(f[1,1])
vlines!(ax, source, ymin=-0.0, ymax=0.0)
vlines!(ax, target, ymin=0.5, ymax=1.0)
vlines!(ax, target .+ (0.005 .* rand(length(target))) .- 0.01, ymin=0.0, ymax=0.5)
f

## Van Rossum Metric; interesting stuff

function spikedist(u, v, τ)
    d = sum(exp(-abs(ui-uj)/τ) for ui in u for uj in u) +
        sum(exp(-abs(vi-vj)/τ) for vi in v for vj in v) -
        2 * sum(exp(-abs(ui-vi)/τ) for ui in u for vi in v)
    return d
end

##
bob = @subset(df, :moth .== "2023-05-25" .&& :poke .== 2)
s1 = bob[bob.unit .== "94", :abstime]
s2 = bob[bob.unit .== "lba", :abstime]

X = exp10.(range(log10(0.0001), stop=log10(0.1), length=20))
dist = [spikedist(s1, s2, x) for x in X]
f = Figure()
ax = Axis(f[1,1], xscale=log10)
lines!(ax, X, dist)
current_figure()

println(X[findmin(dist)[2]])

##
bob = @subset(df, :moth .== "2023-05-25" .&& :poke .== 1)
target = bob[bob.unit .== "lba", :abstime]
source = bob[bob.unit .== "18", :abstime]
sort!(target)
sort!(source)
target, source = convert.(Float64, target), convert.(Float64, source)

include("CoTETE.jl/src/CoTETE.jl")
import .CoTETE
parameters = CoTETE.CoTETEParameters(
            l_x = 3, l_y = 2,
            transform_to_uniform=true, 
            k_global=4,
            num_surrogates=100, 
            use_exclusion_windows=false,
            add_dummy_exclusion_windows=false,
            # sampling_method="jittered_target",
            # jittered_sampling_noise=0.05,
            # num_samples_ratio=4.0,
            auto_find_start_and_num_events=true,
            metric=Cityblock())
CoTETE.estimate_TE_from_event_times(parameters, source, target)

##
xrange, yrange = 2:2:20, 2:2:20
TE = zeros(length(xrange),length(yrange))
for (i,lx) in enumerate(xrange)
    for (j,ly) in enumerate(yrange)
        parameters = CoTETE.CoTETEParameters(
            l_x = lx, l_y = ly,
            transform_to_uniform=false, 
            k_global=4,
            num_surrogates=100, 
            use_exclusion_windows=false,
            add_dummy_exclusion_windows=false,
            # sampling_method="jittered_target",
            # jittered_sampling_noise=0.05,
            # num_samples_ratio=4.0,
            auto_find_start_and_num_events=true,
            metric=Euclidean())

        bob = target .+ (0.005 .* rand(length(target))) .- 0.01
        # target = target[vcat(diff(target), 0) .> 0.04]

        TE[i,j] = CoTETE.estimate_TE_from_event_times(parameters, target, bob)
    end
end

##
fig, ax, hm = heatmap(xrange, yrange, TE)
Colorbar(fig[:, end+1], hm)
ax.xlabel = "Muscle embed length"
ax.ylabel = "Fake muscle embed length"
ax.title = ""
fig

# TE, p, TE_surrogate, locals = CoTETE.estimate_TE_and_p_value_from_event_times(parameters, target, source;
#     return_surrogate_TE_values=true, return_locals=true)

##

muscles = unique(bob[bob.ismuscle, :unit])
neurons = unique(bob[(!).(bob.ismuscle), :unit])
parameters = CoTETE.CoTETEParameters(
    l_x = 3, l_y = 3,
    transform_to_uniform=false, 
    k_global=4, 
    num_surrogates=50, 
    use_exclusion_windows=true,
    num_samples_ratio=4.0,
    auto_find_start_and_num_events=true,
    num_target_events=1000)

muscle_spikes = bob[bob.unit .== "rax", :abstime]
nmTE = Dict{String, Float64}()
mnTE = Dict{String, Float64}()
for (i,n) in enumerate(neurons)
    neuron_spikes = bob[bob.unit .== n, :abstime]
    if length(neuron_spikes) < 100
        continue
    end
    println(n)
    # params, target, source
    #muscle -> neuron
    mnTE[n] = CoTETE.estimate_TE_from_event_times(parameters, neuron_spikes, muscle_spikes)
    #neuron -> muscle
    nmTE[n] = CoTETE.estimate_TE_from_event_times(parameters, muscle_spikes, neuron_spikes)
    println("$(mnTE[n]) vs $(nmTE[n])")
    if mnTE[n] > nmTE[n]
        println("   More likely ascending?")
    else
        println("   More likely descending?")
    end
end
# TE, p = CoTETE.estimate_TE_and_p_value_from_event_times(parameters, target, source)

##
f = Figure()
ax = Axis(f[1,1])
hm = heatmap!(ax, lxrange, lyrange, TE)
Colorbar(f[:, end+1], hm)
current_figure()

##
f = Figure()
ax = Axis(f[1,1])
vlines!(ax, target; ymin=0.0, ymax=0.25)
vlines!(ax, source; ymin=0.3, ymax=0.55)
display(f)

##
function thin_target(source, target, target_rate)
    start_index = 1
    while target[start_index] < source[1]
         start_index += 1
    end
    target = target[start_index:end]
    new_target = Float64[]
    index_of_last_source = 1
    for event in target
        while index_of_last_source < length(source) && source[index_of_last_source + 1] < event
                 index_of_last_source += 1
        end
        distance_to_last_source = event - source[index_of_last_source]
        lambda = 0.5 + 5exp(-50(distance_to_last_source - 0.5)^2) - 5exp(-50(-0.5)^2)
        if rand() < lambda/target_rate
              push!(new_target, event)
        end
    end
    return new_target
end

source = sort(1e4*rand(Int(1e4)))
target = sort(1e4*rand(Int(1e5)))
# append!(target, sort(1e4*rand(Int(1e4))))
# sort!(target)
target = thin_target(source, target, 10)
TE = CoTETE.estimate_TE_from_event_times(CoTETE.CoTETEParameters(l_x = 3, l_y = 3), target, source)


##
dt = DataFrame()
parameters = CoTETE.CoTETEParameters(
    l_x = 5, l_y = 5,
    transform_to_uniform=false,
    k_global=3,
    num_surrogates=50)
for gdf in groupby(df, :moth)
    thismoth = gdf.moth[1]
    println("Moth : $(thismoth)")
    for combo in combinations[thismoth]
        target, source = gdf[gdf.unit .== combo[1], :abstime], gdf[gdf.unit .== combo[2], :abstime]
        sort!(target)
        sort!(source)
        println("$(combo[1]), $(combo[2])")
        if length(target) < 50 || length(source) < 50
            continue
        end
        TEnm = CoTETE.estimate_TE_from_event_times(parameters, target, source)
        TEmn = CoTETE.estimate_TE_from_event_times(parameters, source, target)
        dt = vcat(dt, DataFrame(
            :moth => thismoth,
            :muscle => combo[1],
            :neuron => combo[2], 
            # :p => p,
            :TEnm => TEnm
            :TEmn => TEmn,
        ))
    end
end

@transform!(dt, :MI = :TEmn + :TEnm)

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

## Histograms of neuron spike phase/time
dplot = @pipe df |> 
    @subset(_, (!).(:ismuscle)) |> 
    @transform(_, :unit = ifelse.(:moth .== "2023-05-20", :unit .* "a", :unit .* "b")) |> 
    groupby(_, :unit) |> 
    transform(_, nrow => :nwb) |> 
    @subset(_, :nwb .> 100)
n_units = length(unique(dplot.unit))
n_row = 3
@pipe dplot |> 
    (
    AlgebraOfGraphics.data(_) *
    mapping(:phase => "Neuron spike phase (a.u.)", layout=:unit) *
    histogram(bins=100, normalization=:pdf, datalimits=extrema) *
    visual(alpha=0.6)
    ) |> 
    draw(_, 
        figure=(resolution=(2000, 900),),
        palettes=(layout=vec([(r, c) for r in 1:n_row, c in 1:ceil(Int, n_units/n_row)]),),
        facet=(; linkxaxes=:colwise, linkyaxes=:none),
        axis=(; limits=(nothing, nothing), 
            xticks=([0, 0.5, 1.0], ["0", "0.5", "1"]),
            ygridvisible=false, yticklabelsvisible=false, yticksvisible=false, ylabel="")) #|> 
    # save(joinpath(figsdir, "neuron_phase_hist.png"), _)
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
# group by moth, poke, loop over (muscle, neuron) combinations, calculate precision and make a plot
precFirst = Dict{Tuple{String, String, String}, Dict{Symbol, Any}}()
dt = DataFrame()
for gdf in groupby(df, :moth)
    thismoth = gdf.moth[1]
    for combo in combinations[thismoth]
        println("Moth $thismoth, Muscle: $(combo[1]), Neuron: $(combo[2])")
        X, Y = XY_array_from_dataframe((combo[1], combo[2]), gdf.unit, gdf.time .* 1000, gdf.wb)
        mask = (!).(isnan.(X[:,1]))
        X, Y = X[mask, 1:1], Y[mask, :]
        if size(X,1) <= (1 + size(Y,2)) .|| (size(X,1) < 100)
            continue
        end
        f = Figure()
        ax = Axis(f[1,1], xscale=log10, xlabel="Added noise amplitude (ms)", ylabel="MI (bits)")
        precision_val, zero_noise_MI, sd, MI_curve = precision(X, Y;
            noise=exp10.(range(log10(0.05), stop=log10(100), length=200)),
            repeats=100,
            do_plot=true,
            ax=ax
        )
        text!(ax, 0.2, 0.2, text="# wingbeats = $(size(X,1))", fontsize=24, space=:relative)
        text!(ax, 0.2, 0.1, text="dimensionality = $(size(X,2) + size(Y,2))", fontsize=24, space=:relative)
        ax.title = "Moth $thismoth, Muscle: $(combo[1]), Neuron: $(combo[2])"
        save(joinpath(figsdir, "precision_curves_only_first_new_sort", combo[1], join((thismoth, combo[1], combo[2]), "_") * ".png"), f)

        precFirst[(thismoth, combo[1], combo[2])] = Dict(
            :precision => precision_val,
            :MI => zero_noise_MI,
            :sd => sd,
            :curve => MI_curve)
        dt = vcat(dt, DataFrame(
            :moth => thismoth,
            :muscle => combo[1],
            :neuron => combo[2],
            :precision => precision_val,
            :MI => zero_noise_MI,
            :sd => sd,
            :nwb => size(X,1),
            :dims => size(X,2) + 1
        ))
    end
end

# TODO: Better system that saves parameters I ran this under (like number of repeats)
save(joinpath(analysis_dir, "precision_muscle_first_spike.jld"), "GOV", precFirst)
CSV.write(joinpath(analysis_dir, "precision_muscle_first_spike.csv"), dt)

## Precision for just phasic neurons
dt = @pipe CSV.read(joinpath(analysis_dir, "precision_muscle_first_spike.csv"), DataFrame) |> 
    @subset(_, :precision .!= 0.0) |> 
    @subset(_, (!).(isnan.(:precision)) .|| :MI .!= 0.0) |> 
    @subset(_, :precision .> 1) |> 
    @transform(_, :moth = string.(:moth)) |> 
    @subset(_, :neuron .== 13 .|| :neuron .== 94 .|| :neuron .== 97) |> 
    @subset(_, :precision .< 50)

f = Figure(resolution=(1000,800))
ax = Axis(f[1,1], xlabel="Precision (ms)")
color_dict = Dict(unq => Makie.wong_colors()[i] for (i,unq) in enumerate(unique(dt.neuron)))
hist!(ax, dt.precision, normalization=:probability)
scatter!(ax, dt.precision, rand(length(dt.precision)) .* 0.05 .- 0.07, color=[color_dict[x] for x in dt.neuron])
xlims!(ax, low=0, high=50)
hideydecorations!(ax)
save(joinpath(figsdir, "precision_distribution_phasic_only.png"), f)
current_figure()

## Distributions of zero-noise MI and precision values

"""
Not NaN but still remove:
05-25, lax, 18
05-25, lax, 25
"""
good_neurons = @pipe df |> 
    @subset(_, (!).(:ismuscle) .&& :quality .== "good") |> 
    unique(_.unit) |> 
    parse.(Int, _)

dt = @pipe CSV.read(joinpath(analysis_dir, "precision_muscle_first_spike.csv"), DataFrame) |> 
    @subset(_, :precision .!= 0.0) |> 
    @subset(_, (!).(isnan.(:precision)) .|| :MI .!= 0.0) |> 
    @subset(_, :precision .> 1) |> 
    @transform(_, :moth = string.(:moth)) |> 
    # transform(_, :muscle =>  ByRow(s -> uppercase.(s[2:end])) => :muscle) |> 
    transform(_, :neuron => ByRow(x -> x in good_neurons ? "good" : "mua") => :quality)

f = Figure(resolution=(1000,800))
axtop = Axis(f[1,1])
axmain = Axis(f[2,1], xlabel="Mutual information (bits/ws)", ylabel="Precision (ms)", 
    yscale=Makie.pseudolog10, yticks=[0, 10, 100],
    yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(10))
axright = Axis(f[2,2], yscale=Makie.pseudolog10, yticks=[0, 10, 100],
    yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(10))
linkyaxes!(axmain, axright)
linkxaxes!(axmain, axtop)

for gdf in groupby(dt, :moth)
    scatter!(axmain, gdf.MI, gdf.precision, label=first(gdf.moth))
    # density!(axtop, gdf.MI)
    # density!(axright, gdf.precision, direction=:y)
    # hist!(axright, gdf.precision, direction=:x, 
    #     bins=exp10.(range(log10(1), stop=log10(100), length=40)), normalization=:density)
end
density!(axtop, dt.MI)
density!(axright, dt.precision, direction=:y)
leg = Legend(f[1,2], axmain)
leg.tellheight = true
hideydecorations!(axtop)
hidexdecorations!(axtop, grid=false, minorgrid=false, ticks=false, minorticks=false)
hidexdecorations!(axright)
hideydecorations!(axright, grid=false, minorgrid=false, ticks=false, minorticks=false)
xlims!(axmain, low=0)
ylims!(axmain, low=0)
ylims!(axright, low=0)
# save(joinpath(figsdir, "precision_vs_MI_logScalePrecision.png"), f)
current_figure()



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
    @subset(_, :wbfreq .> 2) |> 
    (
    AlgebraOfGraphics.data(_) *
    mapping(:wbfreq => "Wingbeat Frequency (Hz)", color=:moth => "Moth") * 
    histogram(bins=100, normalization=:pdf, datalimits=extrema) *
    visual(alpha=0.5)
    ) |> 
    draw(_,
        figure=(resolution=(1500, 900),),
        axis=(; limits=((0, 20), (0, nothing)), ylabel="Probability density")) |> 
    save(joinpath(figsdir, "wingbeat_frequency_all.png"), _)
current_figure()

##

bob = @pipe df |> 
    groupby(_, [:moth, :poke, :unit]) |> 
    combine(_, nrow)

jim = @pipe df |> 
    @subset(_, (!).(:ismuscle)) |> 
    groupby(_, [:moth, :poke, :unit, :quality]) |> 
    combine(_, nrow)
