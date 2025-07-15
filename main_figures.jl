using HDF5
using CSV
using NPZ
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


df = DataFrame()
for task in 1:5
    read_run_file!(df, joinpath(data_dir, "2025-07-13_main_single_neurons_PACE_task_$(task).h5"), task)
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
df = @pipe df |> 
    @transform(_, :single = ifelse.(occursin.("-", :muscle), false, true)) |> 
    @transform(_, :muscle = getindex.(Ref(muscle_names_dict), :muscle))
##
# For each neuron/muscle combo find how precision scales, window size at which starts scaling with window size

@pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
transform(_, [:window])


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
@transform(_, :mi = :mi ./ :window, :window = :window .* 1000) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
# combine(_, sdf -> sdf[argmax(sdf.precision), :]) |> 
@subset(_, :mi .> 0) |> 
groupby(_, [:muscle]) |> 
combine(_, :precision => mean, :precision => std, :mi => mean, :mi => std)


## Kinematics has some dependency on window size
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
@pipe df_kine |> 
groupby(_, [:muscle, :moth]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
# combine(_, sdf -> sdf[argmin(sdf.precision), :]) |> 
@transform(_, 
    :mi = :mi ./ :window,
    :single = in.(:muscle, (single_muscles,))) |> 
@transform(_, :isall = (:muscle .== "all") .+ 0) |> 
(AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, color=:muscle, marker=:single, col=:moth) * visual(Scatter)
) |> 
draw(_)#, axis=(; yscale=log10))


## Main plot

dfmain = @pipe df |> 
@transform(_, :mi = :mi ./ :window, :window = :window .* 1000) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
# combine(_, sdf -> sdf[argmin(sdf.precision), :]) |> 
# @transform(_, :mi = :mi ./ :meanrate) |>  # Convert to bits/spike
# @transform(_, :mi = :mi .* :timeactive ./ :nspikes) |> # Alternative bits/spike
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
# @subset(_, :mi .> 0)
@subset(_, :mi .> 10^-1.5)

dfkine = @pipe df_kine |> 
groupby(_, [:muscle, :moth]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
# combine(_, sdf -> sdf[argmin(sdf.precision), :]) |> 
@transform(_, 
    :direction = "kinematics",
    :mi = :mi ./ :window,
    :single = in.(:muscle, (single_muscles,))) |> 
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :mi .> 10^-1.5)

plt1 = AlgebraOfGraphics.data(dfmain) * 
mapping(:mi=>"Mutual Information (bits/s)", :precision_noise=>"Spike timing precision (ms)", 
    # row=:moth, 
    col=:muscle, 
    row=:direction,
    color=:direction
    # color=:label=>nonnumeric
    # color=:embed=>nonnumeric
) * visual(Scatter, alpha=0.4)
plt2 = AlgebraOfGraphics.data(dfkine) * 
mapping(:mi=>"Mutual Information (bits/s)", :precision=>"Spike timing precision (ms)", 
    col=:muscle, 
    row=:direction,
    color=:direction) * visual(Scatter, alpha=0.4)

draw(plt1 + plt2, axis=(; yscale=log10, xscale=log10))#, xscale=log10))


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

## What's up with these imprecise neurons?

@pipe df |> 
select(_, Not([:precision_curve, :precision_levels])) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@subset(_, :precision .> 10^1.5)


## What's up with these super precise neurons?
rows = @pipe df |> 
@transform(_, :mi = :mi ./ :window) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@subset(_, :direction .== "descending", :precision .< 10^0.25, :mi .> 0) |> 
eachrow(_)

case = [rows[1].moth, rows[1].neuron, rows[1].muscle]

f = Figure()
ax = Axis(f[1,1], xscale=log10)
# for row in rows
#     lines!(ax, row.precision_levels, row.precision_curve[2:end])
#     vlines!(ax, row.window .* 1000)
# end
dt = @subset(df, :moth .== case[1], :neuron .== case[2], :muscle .== case[3])
for row in eachrow(dt)
    lines!(ax, row.precision_levels, row.precision_curve[2:end] ./ row.window, color=row.window, colorrange=(0.02, 0.1))
end
f


## Stacked bar plot for each neuron 
categories = ["power", "steering"]
# categories = ["Lpower", "Rpower"]
# categories = ["Lsteering", "Rsteering"]
# categories = ["Lpower", "Lsteering"]
# categories = ["Lsteering", "Lpower", "Rpower", "Rsteering"]
# categories = ["ldvm", "rdvm"]
# categories = single_muscles

ddt = @pipe df |> 
select(_, Not([:precision_curve, :precision_levels])) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
@transform(_, :stack = indexin(:muscle, categories)) |> 
@subset(_, (!).(isnothing.(:stack)), :stack .!= 0) |> 
groupby(_, [:moth, :neuron]) |> 
transform(_, [:mi, :muscle] => ((mi, muscle) -> first(mi[muscle .== categories[1]])) => :mi_sort) |> 
# Arrange fractions, ratios
# @transform(_, :mi = ifelse.(:muscle .== "steering", :mi ./ 6, :mi ./ 4)) |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :mi_total = sum(:mi)) |> 
@transform(_, :mi = :mi ./ :mi_total)
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
            stack=dt.stack, 
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
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
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

## What optimal window sizes were chosen?

@pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window, :precision, row=:moth, col=:muscle, color=:direction) * visual(Scatter)
) |> 
draw(_)



## Look at how window size affected precision, information

dt = @pipe df |> 
@transform(_, :neuron = string.(:neuron) .* :moth) |> 
@transform(_, :mi = :mi ./ :window, :window = :window .* 1000) |> 
@subset(_, :mi .> 0) 

@pipe dt[sortperm(dt.window),:] |> 
# groupby(_, [:moth, :neuron]) |> 
# @transform(_, :maxmi = log.(maximum(:mi))) |> 
# @subset(_, :maxmi .> 0) |> 
@subset(_, :moth .== moths[6]) |> 
# @transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :muscle .== "rax") |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :chosen_precision = :precision[argmax(:mi)]) |> 
@subset(_, :chosen_precision .< 10^1.) |> 
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
function add_intercept_column(x::AbstractVector{T}) where {T}
    mat = similar(x, float(T), (length(x), 2))
    fill!(view(mat, :, 1), 1)
    copyto!(view(mat, :, 2), x)
    return mat
end
# Modified version which plots x and y flipped
function plot_fit_line!(ax, x, y; plotargs...)
    df = DataFrame(x=x, y=y)
    model = lm(@formula(y ~ x), df)
    pred = GLM.predict(model, add_intercept_column(df.x); interval=:confidence, level=0.95)
    inds = sortperm(df.x)
    band!(ax, Point2f.(x[inds], pred.lower[inds]), Point2f.(x[inds], pred.upper[inds]); alpha=0.5, plotargs...)
    lines!(ax, x, pred.prediction; linewidth=3, plotargs...)
    return model
end
function breakpoint(x, y)
    n_breaks = length(x) - 3 - 4 + 1
    slope = zeros(n_breaks, 2)
    intercept = zeros(n_breaks, 2)
    fit = zeros(n_breaks, 3)
    for (i,ind) in enumerate(4:(length(x)-3))
        mod1 = lm(@formula(y ~ x), DataFrame(x=x[1:ind], y=y[1:ind]))
        mod2 = lm(@formula(y ~ x), DataFrame(x=x[ind:end], y=y[ind:end]))
        intercept[i,1] = coef(mod1)[1]
        intercept[i,2] = coef(mod2)[1]
        slope[i,1] = coef(mod1)[2]
        slope[i,2] = coef(mod2)[2]
        fit[i,1] = sum(residuals(mod1).^2)
        fit[i,2] = sum(residuals(mod2).^2)
        fit[i,3] = (fit[i,1] + fit[i,2])#  * -1/(length(x) - 1)
    end
    return slope, intercept, fit
end
function find_scaling_point(x; threshold=0.5, allowed_above=2)
    boolvec = x .< threshold
    if !any(boolvec)
        return 1
    end
    vals, lens = rle(boolvec)
    # Find the first run where vals is false (meaning x >= threshold)
    # and the run length is greater than allowed_above
    cumulative_pos = 0
    for i in eachindex(vals)
        if !vals[i] && lens[i] >= allowed_above
            # Found the change point - return the position where this run starts
            return cumulative_pos + 1
        end
        cumulative_pos += lens[i]
    end
    # No change point found
    return length(x)
end

bob = @pipe dt[sortperm(dt.window),:] |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :chosen_precision = :precision[argmax(:mi)])
# @subset(_, :chosen_precision .< 10^1.)
bob = bob[bob.neuron .== rand(unique(bob.neuron)), :]
bob = bob[bob.muscle .== rand(unique(bob.muscle)), :]
# bob = bob[bob.neuron .== "15.02025-03-21", :]
# bob = bob[bob.muscle .== "lax", :]

# Problem cases: 2025-03-21 neuron 15 muscle rax
# Textbook looks great: 2025-02-25 neuron 24 muscle lax

slope, intercept, fit = breakpoint(bob.window, bob.precision)

ind = find_change(slope[:,1]; threshold=0.5, allowed_above=2)
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