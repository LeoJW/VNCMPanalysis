using HDF5
using CSV
using NPZ
using Statistics
using GLM
using CairoMakie
using GLMakie
using AlgebraOfGraphics
using DataFrames
using DataFramesMeta
using Pipe
using SavitzkyGolay


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

function find_precision_threshold(noise_levels, mi; threshold=0.9)
    # sg_window = 2 * floor(Int, length(noise_levels) / 5) + 1
    # sg_window = sg_window < 2 ? 5 : sg_window
    sg_window = 51
    curve = savitzky_golay(mi[2:end], sg_window, 2; deriv=0).y ./ mi[1]

    # curve = mi ./ mi[1]
    ind = findfirst(curve .< threshold)
    if isnothing(ind)
        return NaN
    else
        return noise_levels[ind]
    end
end

function read_precision_kinematics_file!(df, file, task)
    precision_noise = h5read(joinpath(data_dir, file), "dict_0")
    precision_curves = h5read(joinpath(data_dir, file), "dict_1")
    subsets = h5read(joinpath(data_dir, file), "dict_2")
    mi_subsets = h5read(joinpath(data_dir, file), "dict_3")
    # Construct dataframe
    first_row = split(first(keys(precision_noise)), "_")
    names = vcat(first_row[1:2:end], ["mi", "precision", "precision_curve", "precision_noise", "subset", "mi_subset"])
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat(
        [x ? Float64 : String for x in is_numeric], 
        Float64, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}
    )
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(precision_noise)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        vals[findfirst(names .== "rep")] = task
        push!(thisdf, vcat(
            vals,
            precision_curves[key][1] .* log2(exp(1)), 
            find_precision_threshold(precision_noise[key] .* 1000, precision_curves[key][2:end]),
            [precision_curves[key] .* log2(exp(1))],
            [precision_noise[key] .* 1000],
            [subsets[key]],
            [mi_subsets[key] .* log2(exp(1))]
        ))
    end
    append!(df, thisdf)
end

function read_run_file!(df, file, task)
    precision_levels = h5read(joinpath(data_dir, file), "dict_0")
    precision_curves = h5read(joinpath(data_dir, file), "dict_1")
    params = h5read(joinpath(data_dir, file), "dict_2")
    # Construct dataframe
    first_row = split(first(keys(precision_levels)), "_")
    names = vcat(first_row[1:2:end], ["mi", "precision", "precision_curve", "precision_levels", "embed"])
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat(
        [x ? Float64 : String for x in is_numeric], 
        Float64, Float64, Vector{Float64}, Vector{Float64}, Int
    )
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(precision_levels)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        embed_ind = findfirst(occursin.("embed_dim", params[key]))
        embed = parse(Int, split(params[key][embed_ind], ":")[2])
        push!(thisdf, vcat(
            vals,
            precision_curves[key][1] .* log2(exp(1)), 
            find_precision_threshold(precision_levels[key] .* 1000, precision_curves[key][2:end]),
            [precision_curves[key] .* log2(exp(1))],
            [precision_levels[key] .* 1000],
            embed
        ))
    end
    append!(df, thisdf)
end

function get_neuron_statistics(; moths=moths, duration_thresh=10, fsamp=30000)
    thisdf = DataFrame()
    for moth in moths
        spikes = npzread(joinpath(data_dir, "..", moth * "_data.npz"))
        labels = npzread(joinpath(data_dir, "..", moth * "_labels.npz"))
        bouts = npzread(joinpath(data_dir, "..", moth * "_bouts.npz"))
        # Remove muscles
        for unit in keys(spikes)
            if (!).(occursin(r"[0-9]", unit))
                delete!(spikes, unit)
                delete!(labels, unit)
                delete!(bouts, unit)
            end
        end
        # Pull out stats
        for neuron in keys(spikes)
            # Get firing rate in active periods, total span of time active
            diff_vec = diff(spikes[neuron])
            diff_over_thresh = findall(diff_vec .> (duration_thresh .* fsamp))
            end_inds = vcat(diff_over_thresh, length(spikes[neuron]))
            start_inds = vcat(1, diff_over_thresh .+ 1)
            spike_counts = end_inds - start_inds 
            bout_times = spikes[neuron][end_inds] - spikes[neuron][start_inds]
            
            mask = bout_times .!= 0
            mean_spike_rate = mean(spike_counts[mask] ./ bout_times[mask]) * fsamp
            total_time = sum(bout_times) / fsamp
            # Put as row in dataframe
            append!(thisdf, DataFrame(
                moth=moth,
                neuron=parse(Int, neuron),
                label=labels[neuron],
                nspikes=length(spikes[neuron]),
                meanrate=mean_spike_rate,
                timeactive=total_time
            ))
        end
    end
    return thisdf
end

df = DataFrame()
for task in 1:5
    read_run_file!(df, joinpath(data_dir, "2025-07-07_main_single_neurons_PACE_task_$(task).h5"), task)
end
df_kine = DataFrame()
for task in 0:5
    read_precision_kinematics_file!(df_kine, joinpath(data_dir, "2025-07-02_kinematics_precision_PACE_task_$(task)_hour_09.h5"), task)
end

# Clean up df kinematics
df_kine = @pipe df_kine |> 
    rename(_, :neuron => :muscle)

# Add neuron stats to main dataframe
df_neuronstats = @pipe get_neuron_statistics() |> 
    transform!(_, :moth => ByRow(x -> replace(x, r"_1$" => "-1")) => :moth) |> 
    @transform(_, :label = ifelse.(:label .== 1, "good", "mua"))
df = leftjoin(df, df_neuronstats, on=[:moth, :neuron])

# Add neuron direction stats to main dataframe
df_direction = @pipe CSV.read(joinpath(data_dir, "..", "direction_estimate_stats_all_units.csv"), DataFrame) |> 
    rename(_, :unit => :neuron, Symbol("%>comp") => :prob_descend, Symbol("%<comp") => :prob_ascend) |> 
    transform(_, [:HDIlo, :HDIup] =>
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
single_muscles = [
    "lax", "lba", "lsa", "ldvm", "ldlm", 
    "rdlm", "rdvm", "rsa", "rba", "rax"
]
muscle_names_dict = Dict(
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
df = @pipe df |> 
    @transform(_, :single = ifelse.(occursin.("-", :muscle), false, true)) |> 
    @transform(_, :muscle = getindex.(Ref(muscle_names_dict), :muscle))


## Summary stats table
bob = @pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
# combine(_, sdf -> sdf[argmax(sdf.precision), :]) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
@subset(_, :mi .> 0) |> 
groupby(_, [:muscle]) |> 
combine(_, :precision => mean, :precision => std, :mi => mean, :mi => std)


## Kinematics has strong dependency on window size
@pipe df_kine |> 
groupby(_, [:moth, :muscle, :window]) |> 
combine(_, :mi => mean => :mi, :precision => mean => :precision) |> 
@transform(_, 
    :mi = :mi ./ :window,
    :single = in.(:muscle, (single_muscles,))) |> 
# @subset(_, (!).(:single)) |> 
@subset(_, :single) |> 
(AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, color=:window, col=:moth, row=:muscle) * visual(Scatter)
) |> 
draw(_, axis=(; yscale=log10))

##

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
@pipe df_kine |> 
groupby(_, [:muscle, :moth]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, 
    :mi = :mi ./ :window,
    :single = in.(:muscle, (single_muscles,))) |> 
# transform!(_, :muscle => ByRow(s ->s[2:end]) => :neuron) |> 
# @subset(_, (!).(:single)) |> 
@transform(_, :isall = (:muscle .== "all") .+ 0) |> 
(AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, color=:muscle, marker=:single, col=:moth) * visual(Scatter)
) |> 
draw(_)#, axis=(; yscale=log10))

## Main plot

@pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :mi = :mi ./ :window) |> 
# @transform(_, :mi = :mi ./ :meanrate) |>  # Convert to bits/spike
# @transform(_, :mi = :mi .* :timeactive ./ :nspikes) |> # Alternative bits/spike
@transform(_, :window = :window .* 1000) |> 
# @subset(_, (!).(:single)) |> 
@subset(_, :single) |> 
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :mi .> 10^-1.5) |> 
(AlgebraOfGraphics.data(_) * 
mapping(:mi=>"Mutual Information (bits/s)", :precision=>"Spike timing precision (ms)", 
    # row=:moth, 
    col=:muscle, 
    row=:direction,
    color=:direction
    # color=:label=>nonnumeric
    # color=:embed=>nonnumeric
) * visual(Scatter, alpha=0.4)
) |> 
draw(_, axis=(; yscale=log10, xscale=log10))#, xscale=log10))

## Single muscles, on same plot
@pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
@subset(_, :single) |> 
@subset(_, :mi .> 0) |> 
# groupby(_, [:moth, :muscle]) |> 
# combine(_, :mi => mean => :mi, :precision => mean => :precision) |> 
@transform(_, :side = getindex.(:muscle, 1)) |> 
(AlgebraOfGraphics.data(_) * 
mapping(:mi, :precision, 
    row=:moth, #col=:side,
    color=:side
) * visual(Scatter)
) |> 
draw(_, axis=(; xscale=log10, yscale=log10))

## Stacked bar plot for each neuron 
# categories = ["power", "steering"]
# categories = ["Lpower", "Rpower"]
# categories = ["Lsteering", "Rsteering"]
# categories = ["Rpower", "Rsteering"]
# categories = ["Lsteering", "Lpower", "Rpower", "Rsteering"]
# categories = ["ldvm", "rdvm"]
categories = single_muscles

ddt = @pipe df |> 
select(_, Not([:precision_curve, :precision_levels])) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
@transform(_, :stack = indexin(:muscle, categories)) |> 
@subset(_, (!).(isnothing.(:stack)), :stack .!= 0) |> 
groupby(_, [:moth, :neuron]) |> 
transform(_, [:mi, :muscle] => ((mi, muscle) -> first(mi[muscle .== categories[1]])) => :mi_sort)
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
        sort!(dt, [order(:muscle), order(:label), order(:mi_sort)])
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


## Stacked bar plots for PRECISION

# categories = ["power", "steering"]
# categories = ["Lpower", "Rpower"]
# categories = ["Lsteering", "Rsteering"]
# categories = ["Rpower", "Rsteering"]
# categories = ["Lsteering", "Lpower", "Rpower", "Rsteering"]
# categories = ["ldvm", "rdvm"]
# categories = ["lax", "lba", "lsa"]
categories = single_muscles

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
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
@transform(_, :single = ifelse.(occursin.("-", :muscle), false, true)) |> 
@subset(_, (!).(:single)) |> 
# @subset(_, :single) |> 
@subset(_, :mi .> 0) |> 
(AlgebraOfGraphics.data(_) * 
mapping(:meanrate, :mi,
    row=:moth, col=:muscle, 
    color=:label=>nonnumeric,
) * visual(Scatter)
) |> 
draw(_)#, axis=(; yscale=log10))



## Look at how window size affected precision, information

dt = @pipe df |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
@transform(_, :single = ifelse.(occursin.("-", :muscle), false, true)) |> 
@subset(_, :single) |> 
@subset(_, :mi .> 0) |> 
@subset(_, :muscle .== "ldlm")
@pipe dt[sortperm(dt.window),:] |> 
# groupby(_, [:moth, :neuron]) |> 
# @transform(_, :mi = :mi ./ maximum(:mi)) |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:window, :mi, row=:moth, color=:neuron, group=:neuron=>nonnumeric) * visual(ScatterLines)
) |> 
draw(_)

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