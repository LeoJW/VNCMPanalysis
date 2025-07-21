using HDF5
using CSV
using NPZ
using DelimitedFiles
using Statistics
using StatsBase # Mainly for rle
using GLM
using CairoMakie
using GLMakie
using AlgebraOfGraphics
using DataFrames
using DataFramesMeta
using Pipe
using SavitzkyGolay

include("functions.jl")
include("mainplot_functions.jl")


analysis_dir = @__DIR__
data_dir = joinpath(analysis_dir, "..", "localdata", "estimation_runs")
dark_or_light = "light"
if dark_or_light .== "light"
    set_theme!()
    fig_dir = joinpath(analysis_dir, "figs_light")
else
    set_theme!(theme_dark())
    fig_dir = joinpath(analysis_dir, "figs_dark")
end

poke_side = Dict(
    "2025-02-25" => "L",
    "2025-02-25-1" => "R",
    "2025-03-11" => "R",
    "2025-03-12-1" => "L",
    "2025-03-20" => "R",
    "2025-03-21" => "R"
)
moths = [
    "2025-02-25",
    "2025-02-25_1",
    "2025-03-11",
    "2025-03-12_1",
    "2025-03-20",
    "2025-03-21"
]
single_muscles = [
    "lax", "lba", "lsa", "ldvm", "ldlm", 
    "rdlm", "rdvm", "rsa", "rba", "rax"
]
muscle_names_dict = Dict(
    "lax-lba-lsa-ldvm-ldlm-rdlm-rdvm-rsa-rba-rax" => "all",
    "lax-lba-lsa-rsa-rba-rax" => "steering",
    "lax-lba-lsa" => "Lsteering",
    "rax-rba-rsa" => "Rsteering",
    "ldvm-ldlm-rdlm-rdvm" => "power",
    "ldvm-ldlm" => "Lpower",
    "rdvm-rdlm" => "Rpower"
)
for muscle in single_muscles
    muscle_names_dict[muscle] = muscle
end
muscle_colors = [
    "lax" => "#94D63C", "rax" => "#6A992A",
    "lba" => "#AE3FC3", "rba" => "#7D2D8C",
    "lsa" => "#FFBE24", "rsa" => "#E7AC1E",
    "ldvm"=> "#66AFE6", "rdvm"=> "#2A4A78",
    "ldlm"=> "#E87D7A", "rdlm"=> "#C14434"
]
muscle_colors_dict = Dict(muscle_colors)
fsamp = 30000


df = DataFrame()
# First run on most moths went fine
for task in 2:5
    read_run_file!(df, joinpath(data_dir, "2025-07-13_main_single_neurons_PACE_task_$(task).h5"), task)
end
# Had to re-run first two moths with more tasks as they were SLOW
for task in 0:5
    read_run_file!(df, joinpath(data_dir, "2025-07-16_main_single_neurons_PACE_task_$(task).h5"), task)
end
df_kine = DataFrame()
for task in 1:2
    read_precision_kinematics_file!(df_kine, joinpath(data_dir, "2025-07-15_kinematics_precision_PACE_task_$(task).h5"), task)
end
# for task in 0:5
#     read_precision_kinematics_file!(df_kine, joinpath(data_dir, "2025-07-02_kinematics_precision_PACE_task_$(task)_hour_09.h5"), task)
# end

# Clean up df kinematics
df_kine = @pipe df_kine |> 
    rename(_, :neuron => :muscle) |> 
    @transform!(_, 
        :mi = :mi ./ :window,
        :single = in.(:muscle, (single_muscles,)))

# Add neuron stats to main dataframe
df_neuronstats = @pipe get_neuron_statistics() |> 
    transform!(_, :moth => ByRow(x -> replace(x, r"_1$" => "-1")) => :moth) |> 
    @transform!(_, :label = ifelse.(:label .== 1, "good", "mua"))
df = leftjoin(df, df_neuronstats, on=[:moth, :neuron])

# Add neuron direction stats to main dataframe
df_direction = @pipe CSV.read(joinpath(data_dir, "..", "direction_estimate_stats_all_units.csv"), DataFrame) |> 
    rename(_, :unit => :neuron, Symbol("%>comp") => :prob_descend, Symbol("%<comp") => :prob_ascend) |> 
    transform!(_, [:HDIlo, :HDIup] =>
        ByRow((HDIlo, HDIup) -> 
            HDIlo < 0.5 && HDIup < 0.5 ? "ascending" :
            HDIlo > 0.5 && HDIup > 0.5 ? "descending" :
            "uncertain") =>
        :direction
    ) |> 
    transform!(_, :moth => ByRow(x -> replace(x, r"_1$" => "-1")) => :moth) |> 
    select(_, [:moth, :neuron, :direction, :prob_descend, :prob_ascend])
df = leftjoin(df, df_direction, on=[:moth, :neuron])

# Clean up some aspects of main dataframe
# Rename muscles to more useful names
df = @pipe df |> 
    @transform!(_, :single = ifelse.(occursin.("-", :muscle), false, true)) |> 
    @transform!(_, :muscle = getindex.(Ref(muscle_names_dict), :muscle))

# For each neuron/muscle combo, get:
# Overall peak mi, peak mi within a valid region, and a valid region defined by limited precision scaling
function get_optimal_mi_precision(window, mi, precision)
    sorti = sortperm(window)
    slope = breakpoint(window[sorti], precision[sorti]; start_window=3)
    ind = find_scaling_point(slope[:,1]; threshold=0.4, allowed_above=2)
    ind = ind == 1 ? ind : ind + 3
    
    max_valid_window = repeat([ind], length(window))
    peak_valid_mi = zeros(Bool, length(window))
    peak_mi = zeros(Bool, length(window))
    if ind != 1
        peak_valid_mi[sorti[argmax(mi[sorti][1:ind])]] = true
    end
    peak_mi[argmax(mi)] = true
    return DataFrame(max_valid_window=max_valid_window, peak_mi=peak_mi, peak_valid_mi=peak_valid_mi)
end

# Convert mutual information to bits/s, windows to milliseconds
df = @pipe df |> 
@transform!(_, :mi = :mi ./ :window, :window = :window .* 1000) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
transform!(_, [:window, :mi, :precision_noise] => get_optimal_mi_precision => [:max_valid_window, :peak_mi, :peak_valid_mi])

# Function assumes grouped by moth
function get_proportion_of_active_windows(moth, neuron, window)
    thismoth = replace(moth[1], r"-1$" => "_1")
    spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
    bouts = npzread(joinpath(data_dir, "..", thismoth * "_bouts.npz"))
    unique_neurons = unique(neuron)
    unique_windows = unique(window)
    # Construct dictionary of number of valid windows for each window size, neuron
    frac_active_dict = Dict(n => Dict{Float64, Float64}() for n in unique_neurons)
    for neur in unique_neurons
        neuron_string = string(round(Int, neur))
        diffvec = diff(spikes[neuron_string])
        bout_inds = searchsortedlast.(Ref(spikes[neuron_string]), round.(Int, bouts["ends"]))
        for wind in unique_windows
            wind_samples = (wind ./ 1000 .* 30000)
            inds_above = findall(diffvec .> wind_samples)
            empty_windows = floor.(Int, diffvec[inds_above] ./ wind_samples)
            # Count jumps that have bout changes in them differently
            cross_bout = findall(in(bout_inds), inds_above)
            if !isempty(cross_bout)
                # Get duration of each bout crossing, remove that time from the empty windows count
                for ind in cross_bout
                    # Get which bout ended here
                    bout_ind = findfirst(bout_inds .== inds_above[ind])
                    bout_gap = floor(Int, (bouts["starts"][bout_ind + 1] - bouts["ends"][bout_ind]) / wind_samples)
                    empty_windows[ind] -= bout_gap
                end
            end
            # Add any missing time on start/end to first and last empty_windows
            empty_windows[1] += floor(Int, (spikes[neuron_string][1] - bouts["starts"][1]) ./ wind_samples)
            empty_windows[end] += floor(Int, (spikes[neuron_string][end] - bouts["ends"][end]) ./ wind_samples)
            n_empty_windows = sum(empty_windows)
            n_total_windows = round(Int, sum(bouts["ends"] .- bouts["starts"]) ./ wind_samples)
            # println("empty windows: $(n_empty_windows) total windows: $(n_total_windows)")
            frac_active_dict[neur][wind] = 1 - (n_empty_windows / n_total_windows)
        end
    end
    frac_active_vec = [frac_active_dict[n][w] for (n,w) in zip(neuron, window)]
    return frac_active_vec
end
# Save MI values converted to bits/s of flapping time, rather than bits/s of any activity
df = @pipe df |> 
groupby(_, [:moth]) |> 
transform!(_, [:moth, :neuron, :window] => get_proportion_of_active_windows => :frac_active) |> 
@transform!(_, :mi = :mi .* :frac_active)


## How much do I believe we're picking the right embed_dim?

row = rand(eachrow(df))
f = Figure()
ax = Axis(f[1,1])
embed = [4, 4, 8, 8, 12, 12]
scatter!(ax, embed, row.embed_mi)
scatter!(ax, [4,8,12], [mean(row.embed_mi[embed .== val]) for val in unique(embed)])
f

## Summary stats table
bob = @pipe df |> 
@subset(_, :peak_valid_mi) |> 
@subset(_, :mi .> 0) |> 
groupby(_, [:muscle]) |> 
combine(_, :precision => mean, :precision => std, :mi => mean, :mi => std)


## Kinematics has some dependency on window size
@pipe df_kine |> 
@subset(_, :single) |> 
groupby(_, [:moth, :muscle, :window]) |> 
combine(_, :mi => mean => :mi, :precision => mean => :precision) |> 
(AlgebraOfGraphics.data(_) *
mapping(:window, :mi, color=:window, col=:moth, row=:muscle) * visual(Scatter)
) |> 
draw(_)#, axis=(; yscale=log10))

## Kinematics subset analysis

dt = @pipe df_kine |> 
@subset(_, :rep .== 0) |> 
@transform(_, :single = in.(:muscle, (single_muscles,))) |> 
# @transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
# @subset(_, (!).(:single)) |> 
@subset(_, :single) |> 
flatten(_, [:subset, :mi_subset]) |> 
@transform(_, :mi_subset = :mi_subset ./ :window, :mi = :mi ./ :window) |> 
@groupby(_, [:subset, :window, :moth, :muscle]) |> 
combine(_, :mi_subset => mean => :mi_subset, :mi_subset => std => :mi_subset_std, :mi => first => :mi) |> 
@groupby(_, [:window, :moth, :muscle]) |> 
combine(_) do d
    row = copy(DataFrame(d[1,:]))
    row.mi_subset .= row.mi
    row.subset .= 1
    return vcat(row, d)
end |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:subset, :mi_subset, color=:window=>log, col=:moth, row=:muscle, group=:window=>nonnumeric) * 
visual(ScatterLines)
) |> 
draw(_)

##
muscle_colors = [
    "lax" => "#94D63C", "rax" => "#6A992A",
    "lba" => "#AE3FC3", "rba" => "#7D2D8C",
    "lsa" => "#FFBE24", "rsa" => "#E7AC1E",
    "ldvm"=> "#66AFE6", "rdvm"=> "#2A4A78",
    "ldlm"=> "#E87D7A", "rdlm"=> "#C14434",
    "power"=>:red, "steering"=>:blue, "all"=>:black
]
@pipe df_kine |> 
# @subset(_, :single) |> 
groupby(_, [:moth, :muscle, :window]) |> 
combine(_, :mi => mean => :mi, :precision => mean => :precision, :single => first => :single) |> 
groupby(_, [:muscle, :moth]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
(AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, color=:muscle=>sorter([m[1] for m in muscle_colors]), marker=:single, col=:moth) * 
visual(Scatter, markersize=14)
) |> 
draw(_, scales(Color=(; palette=[m[2] for m in muscle_colors])))#, axis=(; yscale=log10))


## Main plot

dfmain = @pipe df |> 
@subset(_, :peak_valid_mi) |> 
# @transform(_, :mi = :mi ./ :meanrate) |>  # Convert to bits/spike
# @transform(_, :mi = :mi .* :timeactive ./ :nspikes) |> # Alternative bits/spike
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :mi .> 0)

dfkine = @pipe df_kine |> 
groupby(_, [:muscle, :moth]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :direction = "kinematics") |> 
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :mi .> 10^-1.5)

plt1 = AlgebraOfGraphics.data(dfmain) * 
mapping(:mi=>"Mutual Information (bits/s)", :precision_noise=>"Spike timing precision (ms)", 
    # row=:moth, 
    col=:muscle, 
    row=:direction,
    color=:label
    # color=:embed=>nonnumeric
) * visual(Scatter, alpha=0.4)
plt2 = AlgebraOfGraphics.data(dfkine) * 
mapping(:mi=>"Mutual Information (bits/s)", :precision=>"Spike timing precision (ms)", 
    col=:muscle, 
    row=:direction,
    color=:direction) * visual(Scatter, alpha=0.4)

draw(plt1 + plt2, axis=(; yscale=log10))#, xscale=log10))

## Curve of most precise neuron
# ind = argmin(@subset(df, :peak_valid_mi, :mi .> 1).precision_noise)
# row = @subset(df, :peak_valid_mi, :mi .> 1)[ind, :]
f = Figure()
ax = Axis(f[1,1], xscale=log10)
for row in eachrow(@subset(df, :peak_valid_mi, :moth .== "2025-03-11", :neuron .== 3, (!).(:single)))
    lines!(ax, row.precision_levels, row.precision_curve[2:end], label=row.muscle)
end
f[1,2] = Legend(f, ax)
f

##
@pipe df |> 
@subset(_, :mi .> 0) |> 
@subset(_, :peak_valid_mi, :muscle .== "all") |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:nspikes, :mi, color=:embed) *
visual(Scatter)
) |> 
draw(_, axis=(; xscale=log10))

## Single muscles, on same plot
@pipe df |> 
@transform(_, :mi = :mi ./ :window, :window = :window .* 1000) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@subset(_, :single) |> 
@subset(_, :mi .> 0) |> 
@transform(_, :value = :precision .- :window) |>
# groupby(_, [:moth, :muscle]) |> 
# combine(_, :mi => mean => :mi, :precision => mean => :precision) |> 
@transform(_, :side = getindex.(:muscle, 1)) |> 
(AlgebraOfGraphics.data(_) * 
mapping(:precision, :value, 
    row=:moth, #col=:side,
    color=:label
) * visual(Scatter)
) |> 
draw(_, axis=(; xscale=log10))#, yscale=log10))


## What's up with these super precise neurons?
rows = @pipe df |> 
@subset(_, :peak_valid_mi) |> 
@subset(_, :direction .== "descending", :precision_noise .< 10^0.25, :mi .> 0) |> 
eachrow(_)

f = Figure()
ax = Axis(f[1,1], xscale=log10)
for row in rows
    lines!(ax, row.precision_levels, row.precision_noise_curve[2:end] ./ row.window, color=row.window, colorrange=(15, 200))
    lines!(ax, row.precision_levels, row.precision_curve[2:end] ./ row.window, color=row.window, colorrange=(15, 200), linestyle=:dash)
end
f


## Stacked bar plot for each neuron 
# categories = ["power", "steering"]
# categories = ["Lpower", "Rpower"]
# categories = ["Lsteering", "Rsteering"]
# categories = ["Lpower", "Lsteering"]
# categories = ["Lsteering", "Lpower", "Rpower", "Rsteering"]
categories = ["lax", "lba", "lsa", "ldvm", "ldlm"]
# categories = single_muscles

ddt = @pipe df |> 
select(_, Not([:precision_curve, :precision_levels])) |> 
# @subset(_, :peak_valid_mi) |> 
@subset(_, :peak_mi) |> 
@transform(_, :stack = indexin(:muscle, categories)) |> 
@subset(_, (!).(isnothing.(:stack)), :stack .!= 0) |> 
# groupby(_, [:moth, :neuron]) |> 
# @subset(_, reduce(&, [any(:muscle .== x) for x in categories])) |> 
groupby(_, [:moth, :neuron]) |> 
transform(_, [:mi, :muscle] => ((mi, muscle) -> first(mi[muscle .== categories[1]])) => :mi_sort) |> 
# Arrange fractions, ratios
# @transform(_, :mi = ifelse.(:muscle .== "steering", :mi ./ 6, :mi ./ 4)) |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :mi_total = sum(:mi)) #|> 
# @transform(_, :mi = :mi ./ :mi_total)
# @transform(_, :mi_total = sum(:precision_noise)) |> 
# @transform(_, :mi = :precision_noise)
# @transform(_, :mi = :precision_noise ./ :mi_total)
ddt.stack = Vector{Int64}(ddt.stack)

colors = Makie.wong_colors()

f = Figure()
ax = [Axis(f[i,j]) for i in 1:length(unique(df.moth)), j in 1:3]
for (i,moth) in enumerate(unique(df.moth))
    for (j,direct) in enumerate(unique(df.direction))
        dt = @subset(ddt, :moth .== moth, :direction .== direct)
        if nrow(dt) == 0
            continue
        end
        sort!(dt, [order(:muscle), order(:label), order(:mi_total)])
        nlevels = length(unique(dt.muscle))

        barplot!(ax[i,j], repeat(1:nrow(dt)÷nlevels, nlevels), dt.mi,
            dodge=dt.stack, 
            # color=colors[dt.stack]
            color=dt.stack, colorrange=extrema(dt.stack)
        )
        ax[i,j].xticks = (1:nrow(dt)÷nlevels, string.(trunc.(Int, unique(dt.neuron))))
        ax[i,j].title = moth * " poke side: " * poke_side[dt.moth[1]] * " direction: " * direct
    end
end
linkyaxes!(ax...)
colsize!(f.layout, 1, Auto(4))
f

## Power vs steering fraction against things like firing rate
categories = ["Lpower", "Lsteering"]

# ddt = @pipe df |> 
# select(_, Not([:precision_curve, :precision_levels])) |> 
# groupby(_, [:moth, :neuron, :muscle]) |> 
# combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
# @transform(_, :mi = :mi ./ :window) |> 
# @transform(_, :window = :window .* 1000) |> 
# @transform(_, :stack = indexin(:muscle, categories)) |> 
# @subset(_, (!).(isnothing.(:stack)), :stack .!= 0) |> 
# groupby(_, [:moth, :neuron]) |> 
# transform(_, [:mi, :muscle] => ((mi, muscle) -> first(mi[muscle .== categories[1]])) => :mi_sort) |> 
# @transform(_, :frac = )

## Stacked bar plots for PRECISION

# categories = ["power", "steering"]
# categories = ["Lpower", "Rpower"]
# categories = ["Lsteering", "Rsteering"]
# categories = ["Rpower", "Rsteering"]
categories = ["Lsteering", "Lpower", "Rpower", "Rsteering"]
# categories = ["ldvm", "rdvm"]
# categories = ["lax", "lba", "lsa"]
# categories = single_muscles

ddt = @pipe df |> 
select(_, Not([:precision_curve, :precision_levels])) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :stack = indexin(:muscle, categories)) |> 
@subset(_, (!).(isnothing.(:stack)), :stack .!= 0) |> 
groupby(_, [:moth, :neuron]) |> 
transform(_, [:precision, :muscle] => ((x, muscle) -> first(x[muscle .== categories[1]])) => :mi_sort)
ddt.stack = Vector{Int64}(ddt.stack)

colors = Makie.wong_colors()

f = Figure()
ax = [Axis(f[i,j], yscale=log10) for i in 1:length(unique(df.moth)), j in 1:3]
for (i,moth) in enumerate(unique(df.moth))
    for (j,direct) in enumerate(unique(df.direction))
        dt = @subset(ddt, :moth .== moth, :direction .== direct)
        if nrow(dt) == 0
            continue
        end
        sort!(dt, [order(:muscle), order(:label), order(:mi_sort, rev=true)])
        nlevels = length(unique(dt.muscle))
        
        barplot!(ax[i,j], repeat(1:nrow(dt)÷nlevels, nlevels), dt.precision,
            dodge=dt.stack, 
            # color=colors[dt.stack]
            color=dt.stack, colorrange=extrema(dt.stack)
        )
        ax[i,j].xticks = (1:nrow(dt)÷nlevels, string.(trunc.(Int, unique(dt.neuron))))
        ax[i,j].title = moth * " poke side: " * poke_side[dt.moth[1]] * " direction: " * direct
    end
end
linkyaxes!(ax...)
colsize!(f.layout, 1, Auto(4))
f


## Information, precision, against neuron statistics

@pipe df |> 
@subset(_, :peak_valid_mi) |> 
@subset(_, (!).(:single)) |> 
# @subset(_, :single) |> 
@subset(_, :mi .> 0) |> 
(AlgebraOfGraphics.data(_) * 
mapping(:meanrate, :precision,
    row=:moth, col=:muscle, 
    color=:label=>nonnumeric,
) * visual(Scatter)
) |> 
draw(_)#, axis=(; yscale=log10))

## What optimal window sizes were chosen?
using Colors

dt = @pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), 1, 2)) |> 
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :peak_valid_mi)

bin_edges = sort(unique(df.window)) .- (diff(sort(unique(df.window)))[1] / 2)

direction_colors = Dict(d => alphacolor(Makie.wong_colors()[i], 0.5) for (i,d) in enumerate(unique(dt.direction)))

f = Figure()
ax = [Axis(f[i,1]) for i in 1:2]
for (i,gdf) in enumerate(groupby(dt, [:direction, :peak_off_valid]))
    hist!(ax[gdf.peak_off_valid[1]], gdf.window, 
        bins=bin_edges,
        color=direction_colors[gdf.direction[1]], 
        # weights=gdf.mi, 
        normalization=:pdf,
        label=gdf.direction[1]
    )
end
Legend(f[:,2], ax[1])
f

##

bin_edges = range(0, 40, 5)
window_bins = sort(unique(df.window)) .- 5

@pipe df |> 
@subset(_, :nspikes .> 1000) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), true, false)) |> 
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
# @subset(_, :mi .> 0, :muscle .== "all") |> 
@subset(_, :mi .> 0) |> 
@transform(_, :mi_bin = searchsortedlast.(Ref(bin_edges), :mi)) |> 
# @transform(_, :window = searchsortedfirst.(Ref(window_bins), :mi)) |> 
@subset(_, :peak_valid_mi) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window, color=:mi_bin=>nonnumeric, row=:peak_off_valid, col=:direction) * histogram(normalization=:probability, bins=window_bins) * visual(alpha=0.6)
) |> 
draw(_)



## Look at how window size affected precision, information

dt = @pipe df |> 
# @subset(_, :neuron .== 97) |> 
@transform(_, :neuron = string.(:neuron)) |> # .* :moth) |> 
@subset(_, :mi .> 0) 

@pipe dt[sortperm(dt.window),:] |> 
# @subset(_, :moth .== "2025-03-12-1") |> 
@subset(_, :moth .== "2025-03-11", :neuron .== "6.0") |> 
# @transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :muscle .== "all") |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :chosen_precision = :precision[argmax(:mi)]) |> 
@subset(_, :chosen_precision .< 10^1.5) |> 
# @transform(_, :precision = :precision ./ :window, :precision_noise = :precision_noise ./ :window) |> 
stack(_, [:mi, :precision, :precision_noise]) |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:window, :value, 
    row=:variable, 
    col=:neuron=>nonnumeric,
    color=:window
) * visual(ScatterLines) + 
(mapping([0], [1]) * visual(ABLines))
) |> 
draw(_, facet=(; linkyaxes=:rowwise))


## Try breakpoint segmentation
bob = @pipe dt[sortperm(dt.window),:] |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :chosen_precision = :precision[argmax(:mi)])
# @subset(_, :chosen_precision .< 10^1.)
# bob = bob[bob.neuron .== rand(unique(bob.neuron)), :]
# bob = bob[bob.muscle .== rand(unique(bob.muscle)), :]
bob = bob[bob.neuron .== "41.02025-02-25", :]
bob = bob[bob.muscle .== "all", :]

# Problem cases: 2025-03-21 neuron 15 muscle rax
# Textbook looks great: 2025-02-25 neuron 24 muscle lax

slope = breakpoint(bob.window, bob.precision)

ind = find_scaling_point(slope[:,1]; threshold=0.4, allowed_above=2)
ind += 3

f = Figure()
ax = [Axis(f[i,1]) for i in 1:3]
ax[1].title = bob.neuron[1] * " " * bob.muscle[1]
scatterlines!(ax[1], bob.window, bob.mi)
scatterlines!(ax[1], bob.window[ind:end], bob.mi[ind:end], color=:red)
scatterlines!(ax[2], bob.window, bob.precision_noise)
scatterlines!(ax[2], bob.window[ind:end], bob.precision_noise[ind:end], color=:red)
scatterlines!(ax[3], bob.window[4:end-3], slope[:,1])
scatterlines!(ax[3], bob.window[4:end-3], slope[:,2])
# plot_fit_line!(ax[2], bob.window, bob.precision_noise)
# scatterlines!(ax[4], bob.window[3:end-2], fit[:,1])
# scatterlines!(ax[4], bob.window[3:end-2], fit[:,2])
# scatterlines!(ax[4], bob.window[3:end-2], fit[:,3])
linkxaxes!(ax)
f



##

dt = @pipe df |> 
@transform(_, :mi = :mi ./ :window, :window = :window .* 1000) |> 
@subset(_, :mi .> 0, :neuron .== 42, :moth .== moths[6]) 

@pipe dt[sortperm(dt.window),:] |> 
# groupby(_, [:moth, :neuron, :muscle]) |> 
# @transform(_, :chosen_precision = :precision[argmax(:mi)]) |> 
# @subset(_, :chosen_precision .> 10^1.5) |> 
stack(_, [:mi, :precision, :precision_noise]) |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:window, :value, 
    row=:variable, 
    col=:muscle=>nonnumeric,
    color=:window
) * visual(ScatterLines) + 
(mapping([0], [1]) * visual(ABLines))
) |> 
draw(_, facet=(; linkyaxes=:rowwise))


##

dt = @subset(df, :muscle .== "Rsteering", :moth .== "2025-03-21", :neuron .== 42, :window .< 0.05)

f = Figure()
ax = Axis(f[1,1], xscale=log10)
for row in eachrow(dt)
    # lines!(ax, row.precision_levels, row.precision_curve[2:end] ./ row.window, color=row.window, colorrange=(0.015, 0.2))
    # vlines!(ax, row.precision, color=row.window, colorrange=(0.015, 0.2))
    lines!(ax, row.precision_levels, row.precision_curve[2:end] ./ row.window, color=row.window, colorrange=(0.015, 0.05))
    vlines!(ax, row.precision, color=row.window, colorrange=(0.015, 0.05))
end
f
##
f, ax, ln = vlines(spk_neuron[27] ./ 30000, ymin=0.5, ymax=1)
vlines!(ax, spk_muscle["ldlm"] ./ 30000, ymin=0, ymax=0.5)
f

##

f = Figure()

ax = Axis(f[1,1], xscale=log10)

i = rand(1:nrow(df))
mi = df.precision_curve[i] ./ df.precision_curve[i][1]

hlines!(ax, mi[1])
lines!(ax, df.precision_levels[i], mi[2:end])
hlines!(ax, 0.9, color=:black, linestyle=:dash)
vlines!(ax, df.window[i] * 1000)

# sg_window = 2 * floor(Int, length(df.precision_levels[i]) / 10) + 1
# sg_window = sg_window < 2 ? 5 : sg_window
sg_window = 41
curve = savitzky_golay(mi[2:end], sg_window, 2; deriv=0).y #./ mi[1]
lines!(ax, df.precision_levels[i], curve)


f


## -------------------------------- Figure 1 artifacts and details

## Neuron and Muscle raster plots

neurons, muscles, unit_details = get_spikes("2025-03-12_1")
##
embedding = h5read(joinpath(data_dir, "..", "fig_1_embedding_data.h5"), "dict_0")
plot_range = [1101, 1101.5]

fsamp = 30000
# good_units = [unit for (unit, qual) in unit_details["quality"] if qual == "good"]
good_units = [unit for (unit, qual) in unit_details["quality"] if qual != "noise"]
good_muscles = [m for m in single_muscles if length(muscles[m]) > 0]

unit_fr = [mean(diff(neurons[unit][(neurons[unit] .> plot_range[1] * fsamp) .&& (neurons[unit] .< plot_range[2] * fsamp)])) for unit in good_units]
nspikes = [length(neurons[unit]) for unit in good_units]
mask = (!).(isnan.(unit_fr))
sorti = sortperm(DataFrame(a=unit_fr[mask], b=nspikes[mask], copycols=false))

use_xticks = [plot_range[1], plot_range[1] + (plot_range[2] - plot_range[1]) / 2, plot_range[2]]

f = Figure(size=(700,980))
ax = Axis(f[1,1], xticks=(use_xticks, ["0", "0.25", "0.5"]), yticks=([],[]))
axe = Axis(f[2,1], xticks=(use_xticks, ["0", "0.25", "0.5"]), yticks=([],[]))
axm = Axis(f[3,1], xticks=(use_xticks, ["0", "0.25", "0.5"]), yticks=([],[]), xlabel="Time (s)")

seg = 1 / length(good_units[mask][sorti])
for (i,unit) in enumerate(good_units[mask][sorti])
    # Skip some upper rows for empty space
    if (i >= 13) .&& (i <= 13 + 3)
        continue
    end
    col = unit == 97 ? Makie.wong_colors()[2] : :black # 33 also good
    vlines!(ax, neurons[unit] ./ fsamp, ymin=(i-1)*seg, ymax=i*seg, color=col)
end
seg = 1 / length(good_muscles)
for (i,unit) in enumerate(good_muscles)
    vlines!(axm, muscles[unit] ./ fsamp, ymin=(i-1)*seg, ymax=i*seg, color=muscle_colors_dict[unit])
end

mask = (embedding["time"] .> plot_range[1]) .&& (embedding["time"] .< plot_range[2])
for i in 1:size(embedding["X"],1)
    xvals = embedding["X"][i,mask]
    yvals = embedding["Y"][i,mask]
    xvals = (xvals .- mean(xvals)) ./ std(xvals)
    yvals = (yvals .- mean(yvals)) ./ std(yvals)
    lines!(axe, embedding["time"][mask], xvals .+ 6*i, color=Makie.wong_colors()[2])
    lines!(axe, embedding["time"][mask], yvals .+ 6*i, color=Makie.wong_colors()[1])
end

linkxaxes!(ax, axe, axm)
hidedecorations!(ax)
hidedecorations!(axe)
hideydecorations!(axm, ticks=false)
hidespines!(axe)
hidespines!(ax)
hidespines!(axm)
xlims!(ax, plot_range)
xlims!(axm, plot_range)
rowsize!(f.layout, 1, Auto(1.5))
rowsize!(f.layout, 3, Auto(1.2))
display(f)

# save(joinpath(fig_dir, "fig1_spike_data_with_embedding.svg"), f)

## Embedding values for window example
ind = findmin(abs.(embedding["time"] .- 1101.32555))[2]
mask = (embedding["time"] .> plot_range[1]) .&& (embedding["time"] .< plot_range[2])
println((embedding["X"][:,ind] .- mean(embedding["X"][:,mask], dims=2)) ./ std(embedding["X"][:,mask], dims=2))
println((embedding["Y"][:,ind] .- mean(embedding["Y"][:,mask], dims=2)) ./ std(embedding["Y"][:,mask], dims=2))

## Sine grating

x = range(0, 40, 1000)
y = sin.(x)
mat = hcat([y for i in 1:1000]...)
mat = (mat .+ 1) ./ 2
γ = 0.75
mat = mat .^ (1/γ)
f, ax, hm = image(mat)
hidedecorations!(ax)
save("/Users/leo/Desktop/ResearchPhD/VNCMP/paper/fig_assets/sine_grating.png", f)



## -------------------------------- Figure 2: Training networks, selecting embed dim and window size

function apply_letter_label(g, letter_label)
    Label(g[1, 1, TopLeft()], letter_label,
        fontsize = 26,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
    rowgap!(g, Relative(0.02))
    colgap!(g, Relative(0.01))
end

f = Figure()


# Train/test curve
ga = f[1,1] = GridLayout()
ax1 = Axis(f[1,1], xlabel="Epoch", ylabel="Mutual Information (bits/window)")
tt = h5read(joinpath(data_dir, "..", "fig_2_train_test_data.h5"), "dict_0")

epochs = 1:length(tt["train"])
testfilt = savitzky_golay(tt["test"], 151, 2; deriv=0).y .* log2(exp(1))
trainfilt = savitzky_golay(tt["train"], 151, 2; deriv=0).y .* log2(exp(1))
lines!(ax1, epochs, tt["test"] .* log2(exp(1)), color=Makie.wong_colors()[6], linewidth=0.5)
lines!(ax1, epochs, testfilt, color=Makie.wong_colors()[6], linewidth=3)
lines!(ax1, epochs, tt["train"] .* log2(exp(1)), color=Makie.wong_colors()[3], linewidth=0.5)
lines!(ax1, epochs, trainfilt, color=Makie.wong_colors()[3], linewidth=3)
peakind = findmax(testfilt)
vlines!(ax1, peakind[2], ymin=0, ymax=1, linestyle=:dash, color=:black)
# scatter!(ax1, peakind[2], trainfilt[peakind[2]], color=Makie.wong_colors()[3], markersize=15)
scatter!(ax1, peakind[2], testfilt[peakind[2]], color=:black, markersize=16)
scatter!(ax1, peakind[2], testfilt[peakind[2]], color=Makie.wong_colors()[6], markersize=10)
xlims!(ax1, 0, 500)
ax1.title = "Train model"
apply_letter_label(ga, "A")

# Embed dim selection
gb = f[1,2] = GridLayout()
ax2 = Axis(f[1,2], xticks=[4, 8, 12], xlabel="Embedding Dimension", ylabel="I(X,Y) (bits/window)")
row = first(@subset(df, :moth .== "2025-03-11", :neuron .== 6, :peak_valid_mi, :muscle .== "all"))
embed = [4, 4, 8, 8, 12, 12]
mean_embed = [mean(row.embed_mi[embed .== val]) for val in unique(embed)] .* log2(exp(1))
vlines!(ax2, 8, ymin=0, ymax=1, linestyle=:dash, color=:black)
scatter!(ax2, embed, row.embed_mi .* log2(exp(1)))
scatterlines!(ax2, [4,8,12], mean_embed, markersize=20, color=:black)
xlims!(ax2, 2, 14)
linkyaxes!(ax1, ax2)
ylims!(ax2, 0, nothing)
ax2.title = "Find embedding dimensionality"
apply_letter_label(gb, "B")

# Running precision on many different window sizes
gc = f[1,3] = GridLayout()
ax3 = Axis(f[1,3][1,1], 
    xlabel="Spike timing corruption (ms)", ylabel="I(X,Y) (bits/window)", 
    xscale=log10
)
dt = @subset(df, :moth .== "2025-03-11", :neuron .== 6, :muscle .== "all")
dt = dt[sortperm(dt.window), :]
window_lengths = unique(dt.window)
sg_window = 51
for (i,row) in enumerate(eachrow(dt))
    if mod(i, 5) != 0
        continue
    end
    meanmi = row.precision_noise_curve[2:end]
    curve = savitzky_golay(meanmi, sg_window, 2; deriv=0).y ./ meanmi[1]
    ind = findfirst(curve .< 0.9)
    scatter!(ax3, row.precision_levels[ind], row.precision_noise_curve[ind+1],
        color=row.window, colorrange=[window_lengths[1], window_lengths[end]]
    )
    vlines!(ax3, row.precision_levels[ind], ymin=0, ymax=1,
        color=row.window, colorrange=[window_lengths[1], window_lengths[end]],
        linestyle=:dash
    )
    lines!(ax3, row.precision_levels, row.precision_noise_curve[2:end], 
        color=row.window, colorrange=[window_lengths[1], window_lengths[end]]
    )
end
linkyaxes!(ax2, ax3)
ylims!(ax3, 0, nothing)
cb = Colorbar(f[1,3][1,2], limits=[window_lengths[1], window_lengths[end]], 
    label="Window length (ms)"
)
ax3.title = "Train & estimate precision at \n range of window lengths"
apply_letter_label(gc, "C")

# Window size MI and precision scaling

function plot_mi_precision_against_window!(f, ax_coord, dt)
    ax1 = Axis(f[ax_coord...][1,1], ylabel="I(X,Y) (bits/s)")
    ax2 = Axis(f[ax_coord...][2,1], xlabel="Window length (ms)", ylabel="Spike timing precision (ms)")
    # MI axis
    ind = findmax(dt.mi)[2]
    vlines!(ax1, dt.window[ind], ymin=0, ymax=1, linestyle=:dash, color=:black)
    scatter!(ax1, dt.window[ind], dt.mi[ind], 
    color=:black,
    # color=dt.window[ind], colorrange=extrema(dt.window), 
    markersize=14)
    scatterlines!(ax1, dt.window, dt.mi, color=dt.window)
    # Precision axies
    ablines!(ax2, [0], [1], color=:black)
    xextent, yextent = maximum(dt.window) + 10, maximum(dt.precision) + 10
    poly!(ax2, Point2f[(0,0), (xextent, xextent), (xextent, yextent), (0, yextent)],
    color=:grey, alpha=0.2
    )
    slope = breakpoint(dt.window, dt.precision; start_window=3)
    ind = find_scaling_point(slope[:,1]; threshold=0.4, allowed_above=2)
    ind = ind == 1 ? ind : ind + 3
    scatterlines!(ax2, dt.window[1:ind], dt.precision[1:ind], color=dt.window[1:ind])
    scatterlines!(ax2, dt.window[ind:end], dt.precision[ind:end], color=:grey)

    windows = unique(dt.window)
    midwindow = windows[round(Int, length(windows) / 2)]
    text!(ax2, 
        # 0.5, 0.5, 
        midwindow, midwindow + 50,
        text="No spike timing information", 
        rotation=pi/4,
        align=(:center, :baseline),
        markerspace=:data
    )
    
    linkxaxes!(ax1, ax2)
    xlims!(ax2, 0, xextent)
    ylims!(ax2, 0, yextent)

    return ax1, ax2
end

gd = f[1,4] = GridLayout()
ax41, ax42 = plot_mi_precision_against_window!(f, (1,4), dt)
Label(gd[1,1,Top()], "Choose optimal window size", 
    # fontsize = 20, 
    font=:bold, halign=:center)
apply_letter_label(gd, "D")

# Examples of window scaling for a few interesting neurons

display(f)
