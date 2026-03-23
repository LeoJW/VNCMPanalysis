using HDF5
using CSV
using NPZ
using DelimitedFiles
using Arrow
using FileIO
using Statistics
using StatsBase # Mainly for rle
using HypothesisTests
using Clustering
using Graphs # for clique finding
using GLM
using CairoMakie
using GLMakie
using Colors
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
    read_run_file!(df, joinpath(data_dir, "2025-07-13_main_single_neurons_PACE_task_$(task).h5"))
end
# Had to re-run first two moths with more tasks as they were SLOW
for task in 0:5
    read_run_file!(df, joinpath(data_dir, "2025-07-16_main_single_neurons_PACE_task_$(task).h5"))
end
# For some reason two neurons were missing, fix that here
for task in 0:1
    read_run_file!(df, joinpath(data_dir, "2026-02-23_main_single_neurons_PACE_task_$(task).h5"))
end

df_kine = DataFrame()
for task in 0:5
    read_run_file!(
        df_kine, 
        joinpath(data_dir, "2026-01-20_kinematics_muscles_runs", "2026-01-20_kinematics_precision_PACE_task_$(task).h5"), 
        look_for_subset=true
    )
end

df_kine_neur = DataFrame()
for task in 0:5
    read_run_file!(
        df_kine_neur, 
        joinpath(data_dir, "2026-01-21_kinematics_neurons_runs", "2026-01-21_kinematics_neurons_PACE_task_$(task).h5")
    )
end

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
df_kine = @pipe df_kine |> 
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
df_kine = @pipe df_kine |> 
    @transform!(_, :mi = :mi ./ :window, :window = :window .* 1000) |> 
    groupby(_, [:moth, :muscle]) |> 
    transform!(_, [:window, :mi, :precision_noise] => get_optimal_mi_precision => [:max_valid_window, :peak_mi, :peak_valid_mi])
df_kine_neur = @pipe df_kine_neur |> 
    @transform!(_, :mi = :mi ./ :window, :window = :window .* 1000) |> 
    groupby(_, [:moth, :neuron]) |> 
    transform!(_, [:window, :mi, :precision_noise] => get_optimal_mi_precision => [:max_valid_window, :peak_mi, :peak_valid_mi])

# Save MI values converted to bits/s of flapping time, rather than bits/s of any activity
df = @pipe df |> 
groupby(_, [:moth]) |> 
transform!(_, [:moth, :neuron, :window] => get_proportion_of_active_windows => :frac_active) |> 
@transform!(_, :mi = :mi .* :frac_active)
df_kine_neur = @pipe df_kine_neur |> 
groupby(_, [:moth]) |> 
transform!(_, [:moth, :neuron, :window] => get_proportion_of_active_windows_kine => :frac_active) |> 
@transform!(_, :mi = :mi .* :frac_active)


## How much do I believe we're picking the right embed_dim?

row = rand(eachrow(df_kine_neur))
f = Figure()
ax = Axis(f[1,1])
embed = repeat([4,8,12], 2)
vlines!(ax, row.embed)
scatter!(ax, embed, row.embed_mi)
scatter!(ax, [4,8,12], [mean(row.embed_mi[embed .== val]) for val in unique(embed)])
f

## Look at some window size curves for kinematics
dt = @pipe df_kine_neur |> 
@subset(_, :neuron .∈ Ref(rand(unique(df_kine_neur.neuron), 10))) |> 
@subset(_, :mi .> 0) 

@pipe dt[sortperm(dt.window),:] |> 
@subset(_, :moth .== "2025-02-25-1") |> 
# @transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
# @subset(_, :muscle .== "all") |> 
# groupby(_, [:moth, :muscle]) |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :chosen_precision = :precision[argmax(:mi)]) |> 
@subset(_, :chosen_precision .< 10^1.5) |> 
# @transform(_, :precision = :precision ./ :window, :precision_noise = :precision_noise ./ :window) |> 
stack(_, [:mi, :precision]) |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:window, :value, 
    row=:variable, 
    # col=:muscle=>nonnumeric,
    col=:neuron=>nonnumeric,
    color=:window
) * visual(ScatterLines) + 
(mapping([0], [1]) * visual(ABLines))
) |> 
draw(_, facet=(; linkyaxes=:rowwise))

## Look at some window size curves for muscles to kinematics
dt = @pipe df_kine |> 
@subset(_, :mi .> 0) 

@pipe dt[sortperm(dt.window),:] |> 
# @transform(_, :precision = :precision ./ :window, :precision_noise = :precision_noise ./ :window) |> 
stack(_, [:mi, :precision]) |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:window, :value,
    row=:moth, 
    group=:variable,
    col=:muscle=>nonnumeric,
    color=:window
) * visual(ScatterLines) + 
(mapping([0], [1]) * visual(ABLines))
) |> 
draw(_, facet=(; linkyaxes=:rowwise))



## Summary stats table
bob = @pipe df |> 
@subset(_, :peak_valid_mi) |> 
@subset(_, :mi .> 0) |> 
groupby(_, [:muscle]) |> 
combine(_, :precision => mean, :precision => std, :mi => mean, :mi => std)


## ---------------- Are there neurons more informative of specific muscles than the whole MP?
@pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi, (:neuron .!= 53) .&& (:moth .== "2025-03-12-1")) |> 
groupby(_, [:moth, :neuron]) |> 
transform!(_, [:mi, :muscle] => ((x,y) -> repeat([x[findfirst(y .== "all")]], length(x))) => :allMI) |> 
@transform(_, :mi_ratio = :mi ./ :allMI) |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:allMI, :mi_ratio, color=:direction, layout=:muscle) * 
visual(Scatter) #+ mapping([0],[1]) * visual(ABLines, color=:black)
) |> 
draw(_)

## ---------------- Information and precision of neurons to kinematics vs motor program
df_neuron_to_MP = @pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi, :mi .> 0)

# Mutual information
@pipe df_kine_neur |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi, :mi .> 0) |> 
leftjoin(_, df_neuron_to_MP, on=[:moth, :neuron], renamecols=""=>"_toMP") |> 
# @subset(_, :muscle_toMP .== "all") |> 
@transform(_, :diff = :mi_toMP .- :mi) |> 
(
AlgebraOfGraphics.data(_) * 
# mapping(:mi_toMP=>"Information to MP", :mi=>"Information to kinematics", color=:direction_toMP, layout=:muscle_toMP) * 
mapping(:mi_toMP=>"Information to MP", :diff=>"I(X;Y) - I(X;Z)", color=:direction_toMP, layout=:muscle_toMP) * 
visual(Scatter) + mapping([0],[1]) * visual(ABLines, color=:black)
) |> 
draw(_)#, axis=(; xscale=log10, yscale=log10))
# @pipe df_kine_neur |> 
# @subset(_, :peak_mi, :mi .> 0) |> 
# leftjoin(_, df_neuron_to_MP, on=[:moth, :neuron], renamecols=""=>"_toMP") |> 
# @transform(_, :diff = :mi_toMP .- :mi) |> 
# (
# AlgebraOfGraphics.data(_) * 
# mapping(:mi_toMP, :diff, layout=:muscle_toMP) * 
# visual(Scatter)
# ) |> 
# draw(_)

## Precision
df_neuron_to_MP = @pipe df |> 
@subset(_, :peak_valid_mi, :mi .> 0)

dt = @pipe df_kine_neur |> 
@subset(_, :peak_valid_mi, :mi .> 0) |> 
leftjoin(_, df_neuron_to_MP, on=[:moth, :neuron], renamecols=""=>"_toMP") |> 
@subset(_, (!).(ismissing.(:mi_toMP)), :muscle_toMP .== "all") |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:precision_toMP=>"Precision to MP", :precision=>"Precision to kinematics", color=:direction_toMP, layout=:muscle_toMP) * 
visual(Scatter) + mapping([0],[1]) * visual(ABLines, color=:black)
) |> 
draw(_)

## Optimal window size
df_neuron_to_MP = @pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi, :mi .> 0)

@pipe df_kine_neur |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi, :mi .> 0) |> 
leftjoin(_, df_neuron_to_MP, on=[:moth, :neuron], renamecols=""=>"_toMP") |> 
@subset(_, :muscle_toMP .== "all") |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:window_toMP=>"Optimal window size to MP", :window=>"Optimal window size to kinematics", color=:direction_toMP, layout=:muscle_toMP) * 
visual(Scatter) + mapping([0],[1]) * visual(ABLines, color=:black)
) |> 
draw(_)

##
df_neuron_to_MP = @pipe df |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi, :mi .> 0, :muscle .== "all")

dt = @pipe df_kine_neur |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi, :mi .> 0) |> 
leftjoin(_, df_neuron_to_MP, on=[:moth, :neuron], renamecols=""=>"_toMP")

f = Figure()
axtoph = Axis(f[1,1])
ax = Axis(f[2,1], xlabel="Optimal window size to MP (ms)", ylabel="Optimal window size to kinematics (ms)")
axrighth = Axis(f[2,2])

linkyaxes!(ax, axrighth)
linkxaxes!(ax, axtoph)

n2m_window_bins = sort(unique(df.window))[1:2:end] .- diff(sort(unique(df.window)))[1]
n2k_window_bins = sort(unique(df_kine_neur.window))[1:2:end] .- diff(sort(unique(df_kine_neur.window)))[1]

scatter!(ax, dt.window_toMP .+ (rand(nrow(dt)) .- 0.5) .* 3, dt.window .+ (rand(nrow(dt)) .- 0.5) .* 3)
hist!(axtoph, dt.window_toMP, bins=n2m_window_bins)
hist!(axrighth, dt.window, bins=n2k_window_bins, direction=:x)
ablines!(ax, 0, 1, color=:black)
hidedecorations!(axtoph)
hidedecorations!(axrighth)
colsize!(f.layout, 2, Relative(1/4))
rowsize!(f.layout, 1, Relative(1/4))

display(f)


## Kinematics subset analysis

dt = @pipe df_kine |> 
# @subset(_, :peak_valid_mi) |> 
# @transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
# @subset(_, (!).(:single)) |> 
@subset(_, :single) |> 
flatten(_, [:subset, :mi_subset]) |> 
# @transform(_, :mi_subset = :mi_subset ./ :window) |> 
@groupby(_, [:subset, :window, :moth, :muscle]) |> 
combine(_, :mi_subset => mean => :mi_subset, :mi_subset => std => :mi_subset_std, :mi => first => :mi) |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:subset, :mi_subset, color=:window=>log, col=:moth, row=:muscle, group=:window=>nonnumeric) * 
visual(ScatterLines)
) |> 
draw(_)

##

dkn = @pipe df_kine_neur |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi)

@pipe df |> 
@subset(_, :peak_mi, :moth .∈ Ref(["2025-02-25", "2025-02-25-1"])) |> 
leftjoin(_, @subset(df_kine, :peak_mi), on=[:muscle, :moth], renamecols=""=>"_YZ") |> 
leftjoin(_, dkn, on=[:neuron, :moth], renamecols=""=>"_XZ") |> 
@transform(_, :mi_XY = :mi) |> 
@transform(_, :n2m_loss = :mi_XY .- :mi_XZ) |> 
@transform(_, :m2k_loss = :mi_YZ .- :mi_XZ) |> 
@subset(_, (!).(ismissing.(:mi_XZ))) |> 
# stack(_, [:mi_XY, :mi_XZ, :mi_YZ]) |> 
(
AlgebraOfGraphics.data(_) * 
# mapping(:variable, :value, color=:muscle, group=:neuron=>nonnumeric, layout=:muscle) * visual(ScatterLines) 
mapping(:mi_XY, :mi_XZ, color=:muscle, group=:neuron=>nonnumeric, layout=:muscle) * visual(Scatter) + 
mapping([0],[1]) * visual(ABLines, color=:black)
) |> 
draw(_)

##
mgroups = ["Lpower", "Lsteering", "Rsteering", "Rpower", "steering", "power"]
@pipe df_kine |> 
@subset(_, :peak_valid_mi) |> 
@transform(_, :muscle_group = ifelse.(:muscle .∈ Ref(mgroups), "group", "single/all")) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, color=:muscle_group, marker=:muscle) * visual(Scatter)
) |> 
draw(_, axis=(; limits=(nothing, (0, 34))))


## ---------------- Main plot

dfmain = @pipe df |> 
@subset(_, :peak_valid_mi) |> 
# @transform(_, :mi = :mi ./ :meanrate) |>  # Convert to bits/spike
# @transform(_, :mi = :mi .* :timeactive ./ :nspikes) |> # Alternative bits/spike
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :mi .> 0, :muscle .== "all" .|| :muscle .== "single", :nspikes .> 1000) |> 
@subset(_, :muscle .== "single")

dfkine = @pipe df_kine |> 
groupby(_, [:muscle, :moth]) |> 
combine(_, sdf -> sdf[argmax(sdf.mi), :]) |> 
@transform(_, :direction = "kinematics") |> 
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :mi .> 10^-1.5)

plt1 = AlgebraOfGraphics.data(dfmain) * 
mapping(:mi=>"Mutual Information (bits/s)", :precision=>"Spike timing precision (ms)", 
    # row=:moth, 
    col=:muscle, 
    color=:direction,
    # color=:label
    # color=:embed=>nonnumeric
) * visual(Scatter, alpha=0.4)
plt2 = AlgebraOfGraphics.data(dfkine) * 
mapping(:mi=>"Mutual Information (bits/s)", :precision=>"Spike timing precision (ms)", 
    col=:muscle, 
    row=:direction,
    color=:direction) * visual(Scatter, alpha=0.4)

# draw(plt1 + plt2, axis=(; yscale=log10))#, xscale=log10))
draw(plt1, axis=(; yscale=log10))


## ------------------ Main plot, but kinematics

dfmain = @pipe df |> 
@subset(_, :peak_valid_mi) |> 
# @transform(_, :mi = :mi ./ :meanrate) |>  # Convert to bits/spike
# @transform(_, :mi = :mi .* :timeactive ./ :nspikes) |> # Alternative bits/spike
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :mi .> 0, :nspikes .> 1000, :muscle .== "all")

dfkine = @pipe df_kine |> 
@subset(_, :peak_valid_mi) |> 
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@transform(_, :direction = "kinematics_" .* :muscle) |> 
@subset(_, :mi .> 0)

plt1 = AlgebraOfGraphics.data(dfmain) * 
mapping(:mi=>"Mutual Information (bits/s)", :precision=>"Spike timing precision (ms)", 
    color=:direction,
) * visual(Scatter)
plt2 = AlgebraOfGraphics.data(dfkine) * 
mapping(:mi=>"Mutual Information (bits/s)", :precision=>"Spike timing precision (ms)", 
    color=:direction) * visual(Scatter)

draw(plt1 + plt2,)# axis=(; xscale=log10))
# draw(plt1, axis=(; yscale=log10))


## Stacked bar plot for each neuron 
# categories = ["power", "steering"]
# categories = ["Lpower", "Rpower"]
# categories = ["Lsteering", "Rsteering"]
# categories = ["Lpower", "Lsteering"]
# categories = ["Lsteering", "Lpower", "Rpower", "Rsteering"]
# categories = ["lax", "lba", "lsa", "ldvm", "ldlm"]
# categories = ["lax", "lba", "lsa", "ldvm", "ldlm", "rdlm", "rdvm", "rsa", "rba"]
categories = single_muscles

ddt = @pipe df |> 
select(_, Not([:precision_curve, :precision_levels])) |> 
# @subset(_, :peak_valid_mi) |> 
@subset(_, :peak_mi) |> 
# groupby(_, [:moth, :neuron]) |> 
# transform!(_, [:mi, :muscle] => ((mi,muscle) -> begin
#     ind = findfirst(muscle .== "Lsteering")
#     sendval = isnothing(ind) ? 0.0 : mi[ind]
#     return repeat([sendval], length(mi))
#     end) => :allMI) |> 
@transform(_, :stack = indexin(:muscle, categories)) |> 
@subset(_, (!).(isnothing.(:stack)), :stack .!= 0) |> 
# groupby(_, [:moth, :neuron]) |> 
# @subset(_, reduce(&, [any(:muscle .== x) for x in categories])) |> 
# groupby(_, [:moth, :neuron]) |> 
# transform(_, [:mi, :muscle] => ((mi, muscle) -> first(mi[muscle .== categories[1]])) => :mi_sort) |> 
# Arrange fractions, ratios
# @transform(_, :mi = ifelse.(:muscle .== "steering", :mi ./ 6, :mi ./ 4)) |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :mi_total = sum(:mi)) #|> 
# @transform(_, :mi = :allMI .- :mi)
# @transform(_, :mi_total = sum(:precision_noise)) |> 
# @transform(_, :mi = :precision_noise)
# @transform(_, :mi = :precision_noise ./ :mi_total)
ddt.stack = Vector{Int64}(ddt.stack)

if "rax" in categories
    bob = @pipe ddt |> 
        @subset(_, :moth .== "2025-03-12-1", :peak_mi, :direction .== "descending", :muscle .== "rax")
    row = rand(eachrow(bob))
    row.mi = 0
    push!(ddt, row)
end

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

## -------------------------------- Figure 1 artifacts and details

## Neuron and Muscle raster plots
thismoth = "2025-02-25_1"
neurons, muscles, unit_details = get_spikes(thismoth)
kine = npzread(joinpath(data_dir, "..", thismoth * "_kinematics.npz"))

##
embedding = h5read(joinpath(data_dir, "..", "fig_1_embedding_data.h5"), "dict_0")
plot_range = [700, 700.5] #[1101, 1101.5]

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
axe = Axis(f[2,1])#, xticks=(use_xticks, ["0", "0.25", "0.5"]), yticks=([],[]))
axm = Axis(f[3,1], xticks=(use_xticks, ["0", "0.25", "0.5"]), yticks=([],[]))
axk = Axis(f[4,1], xticks=(use_xticks, ["0", "0.25", "0.5"]), yticks=([],[]), xlabel="Time (s)")

seg = 1 / length(good_units[mask][sorti])
for (i,unit) in enumerate(good_units[mask][sorti])
    # Skip some upper rows for empty space
    # if (i >= 13) .&& (i <= 13 + 3)
    #     continue
    # end
    col = unit == 97 ? Makie.wong_colors()[2] : :black # 33 also good
    vlines!(ax, neurons[unit] ./ fsamp, ymin=(i-1)*seg, ymax=i*seg, color=col)
end
seg = 1 / length(good_muscles)
for (i,unit) in enumerate(good_muscles)
    vlines!(axm, muscles[unit] ./ fsamp, ymin=(i-1)*seg, ymax=i*seg, color=muscle_colors_dict[unit])
end
for (i,angle) in enumerate(["Lphi", "Ltheta", "Lalpha", "Rphi", "Rtheta", "Ralpha"])
    mask = (kine["index"] .> fsamp * plot_range[1]) .&& (kine["index"] .< fsamp * plot_range[2])
    lines!(axk, kine["index"] ./ fsamp, kine[angle] ./ (maximum(kine[angle][mask]) - minimum(kine[angle][mask])) .+ i * 2)
end

# mask = (embedding["time"] .> plot_range[1]) .&& (embedding["time"] .< plot_range[2])
mask = fill(true, length(embedding["time"]))
for i in 1:3
    xvals = embedding["X"][i,mask]
    yvals = embedding["Y"][i,mask]
    xvals = (xvals .- mean(xvals)) ./ std(xvals)
    yvals = (yvals .- mean(yvals)) ./ std(yvals)
    lines!(axe, embedding["time"][mask], xvals .+ 6*i, color=Makie.wong_colors()[2])
    lines!(axe, embedding["time"][mask], yvals .+ 6*i, color=Makie.wong_colors()[1])
end
ylims!(axe, 2, 22)

linkxaxes!(ax, axe, axm, axk)
hidedecorations!(ax)
hidexdecorations!(axm)
hidedecorations!(axe)
hideydecorations!(axm)
hideydecorations!(axk)
hidespines!(axe)
hidespines!(ax)
hidespines!(axm)
hidespines!(axk)
xlims!(ax, plot_range)
xlims!(axm, plot_range)
xlims!(axk, plot_range)
rowsize!(f.layout, 1, Auto(1.5))
rowsize!(f.layout, 3, Auto(1.2))
rowgap!(f.layout, 0)
display(f)

# save(joinpath(fig_dir, "fig1_spike_data_with_embedding.svg"), f)

##
# discrete_vec = zeros(muscles["ldlm"][end]+10)
# discrete_vec[muscles["ldlm"]] .= 1
spec = spectrogram(discrete_vec, fsamp*8, round(Int,fsamp), fs=fsamp)

# mat = 10 .* log10.(power(spec))
fhm, ax, hm = heatmap(time(spec), freq(spec)[1:2000], transpose(power(spec)[1:2000,:]))

## Small subplots of latents against each other

function embedding_fig()
    ind = findmin(abs.(embedding["time"] .- 1101.32555))[2]
    use_shift = embedding["shift_indices"][ind]
    mask = embedding["shift_indices"] .== use_shift

    xcol, ycol = Makie.wong_colors()[1], Makie.wong_colors()[2]
    f = Figure()
    ax = []
    
    ax = [
        Axis(f[1,1], xlabel=L"Z_{Y_1}", ylabel=L"Z_{X_1}", xlabelcolor=xcol, ylabelcolor=ycol,xlabelsize=30, ylabelsize=30),
        Axis(f[1,2], xlabel=L"Z_{Y_2}", ylabel=L"Z_{X_2}", xlabelcolor=xcol, ylabelcolor=ycol,xlabelsize=30, ylabelsize=30),
        Axis(f[1,3], xlabel=L"Z_{Y_3}", ylabel=L"Z_{X_3}", xlabelcolor=xcol, ylabelcolor=ycol,xlabelsize=30, ylabelsize=30)
    ]
    for i in 1:3
        scatter!(ax[i], embedding["X"][i,mask], embedding["Y"][i,mask], alpha=0.3)
        hidedecorations!(ax[i], ticks=false, label=false)
    end
    ylims!(ax[1], nothing, 10)
    f
end

with_theme(embedding_fig, theme_minimal())
# embedding_fig()

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
CairoMakie.activate!()

f = Figure(size=(1500, 900))

# Add a shaded box around panels A-D
box_ad = Box(
    f[1, 1:4, Makie.GridLayoutBase.Outer()],
    alignmode = Outside(-10, -15, -12, -10),
    cornerradius = 8,
    color = (:lightblue, 0.15),
    strokecolor = (:steelblue, 0.8),
    strokewidth = 2,
)
# Move the box to the background so it doesn't cover the plots
Makie.translate!(box_ad.blockscene, 0, 0, -200)

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
scatter!(ax1, peakind[2], testfilt[peakind[2]], color=:black, markersize=16)
scatter!(ax1, peakind[2], testfilt[peakind[2]], color=Makie.wong_colors()[6], markersize=10)

text!(ax1, 0.6, 0.7, text="Training set", 
    font=:bold, color=Makie.wong_colors()[3], space=:relative
)
text!(ax1, 0.6, 0.24, text="Test set", 
    font=:bold, color=Makie.wong_colors()[6], space=:relative
)
xlims!(ax1, 0, 500)
ax1.title = "Train model"
apply_letter_label(ga, "A")

# Embed dim selection
gb = f[1,2] = GridLayout()
ax2 = Axis(f[1,2], xticks=[0, 4, 8, 12], xlabel="Embedding Dimension", ylabel="I(X,Y) (bits/window)")
row = first(@subset(df, :moth .== "2025-03-11", :neuron .== 6, :peak_valid_mi, :muscle .== "all"))
embed = [4, 4, 8, 8, 12, 12]
mean_embed = [mean(row.embed_mi[embed .== val]) for val in unique(embed)] .* log2(exp(1))
vlines!(ax2, 8, ymin=0, ymax=1, linestyle=:dash, color=:black)
scatter!(ax2, embed, row.embed_mi .* log2(exp(1)))
scatterlines!(ax2, [4,8,12], mean_embed, markersize=20, color=:black)
xlims!(ax2, 0, 14)
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

function plot_mi_precision_against_window!(f, ax_coord, dt; xextent=nothing, yextent=nothing, doylabel=true)
    ax1 = Axis(f[ax_coord...][1,1])
    ax2 = Axis(f[ax_coord...][2,1], xlabel="Window length (ms)")
    if doylabel
        ax1.ylabel = "I(X,Y) (bits/s)"
        ax2.ylabel = "Spike timing precision (ms)"
    end
    text!(ax2, 
        Point2f(0.05, 0.8), 
        text="No spike timing \n information", 
        align=(:left, :center),
        space=:relative, 
        color=:grey
    )
    # Chosen precision point
    mi_ind = findmax(dt.mi)[2]
    vlines!(ax1, dt.window[mi_ind], ymin=0, ymax=1, linestyle=:dash, color=:black)
    scatter!(ax1, dt.window[mi_ind], dt.mi[mi_ind], color=:black, markersize=14)
    scatter!(ax2, dt.window[mi_ind], dt.precision[mi_ind], color=:black, markersize=14)
    vlines!(ax2, dt.window[mi_ind], ymin=0, ymax=1, linestyle=:dash, color=:black)
    # MI axis
    scatterlines!(ax1, dt.window, dt.mi, color=dt.window)
    # Precision axis
    ablines!(ax2, [0], [1], color=:black)
    xextent, yextent = maximum(dt.window) + 10, maximum(dt.precision[(!).(isnan.(dt.precision))]) + 10
    poly!(ax2, Point2f[(0,0), (1000, 1000), (0, 1000)],
    color=:grey, alpha=0.2
    )
    slope = breakpoint(dt.window, dt.precision; start_window=3)
    ind = find_scaling_point(slope[:,1]; threshold=0.4, allowed_above=2)
    ind = ind == 1 ? ind : ind + 3
    scatterlines!(ax2, dt.window[1:ind], dt.precision[1:ind], color=dt.window[1:ind])
    scatterlines!(ax2, dt.window[ind:end], dt.precision[ind:end], color=:grey)
    
    linkxaxes!(ax1, ax2)
    xlims!(ax2, 0, xextent)
    ylims!(ax2, 0, yextent)

    return ax1, ax2
end

gd = f[1,4] = GridLayout()
apply_letter_label(gd, "D")
ax41, ax42 = plot_mi_precision_against_window!(f, (1,4), dt)
Label(gd[1,1,Top()], "Choose optimal window size", 
    font=:bold, halign=:center)

# Examples of window scaling for a few interesting neurons
ge = f[2,1:2] = GridLayout()
apply_letter_label(ge, "E")
# 2025-02-25-1 18
dt = @subset(df, :moth .== "2025-02-25-1", :neuron .== 18, :muscle .== "all")
dt = dt[sortperm(dt.window),:]
ax51, ax52 = plot_mi_precision_against_window!(ge, (1,1), dt)
ax51.title = "Moth 2, Neuron 18"
# 2025-03-20 4
dt = @subset(df, :moth .== "2025-03-20", :neuron .== 4, :muscle .== "all")
dt = dt[sortperm(dt.window),:]
ax61, ax62 = plot_mi_precision_against_window!(ge, (1,2), dt; doylabel=false)
ax61.title = "Moth 5, Neuron 4"
# 2025-03-21 42
dt = @subset(df, :moth .== "2025-03-21", :neuron .== 42, :muscle .== "all")
dt = dt[sortperm(dt.window),:]
ax71, ax72 = plot_mi_precision_against_window!(ge, (1,3), dt; doylabel=false)
ax71.title = "Moth 6, Neuron 42"
linkxaxes!(ax51, ax61, ax71, ax52, ax62, ax72)
linkyaxes!(ax51, ax61, ax71)
linkyaxes!(ax52, ax62, ax72)

# Optimal window sizes chosen
gf = f[2,3:end] = GridLayout()
apply_letter_label(gf, "F")

ax8 = Axis(gf[1,1], 
    xlabel="Optimal window size (ms)", ylabel="Probability Density",
    xticks=([0,1/25 * 1000,50,100,150], ["0", "", "50", "100", "150"]),
    xlabelsize=18, ylabelsize=18, xticklabelsize=15, yticklabelsize=15
)
window_bins = sort(unique(df.window))[1:2:end] .- diff(sort(unique(df.window)))[1]
dt = @pipe df |> 
    @transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
    @subset(_, :mi .> 0, :peak_mi, :muscle .== "all")
hist!(ax8, dt.window, bins = window_bins, normalization=:pdf)
vlines!(ax8, 1/25 * 1000, ymin=0, ymax=1, color=:black)
annotation!(ax8, 80, 0.02, 1/25*1000, 0.02, 
    text="25 Hz \n (Typical wingbeat frequency)", 
    labelspace=:data, 
    align=(:left, :center),
    justification=:center,
    path = Ann.Paths.Arc(0.3),
    style=Makie.Ann.Styles.LineArrow(),
    font=:bold, fontsize=18
)
ylims!(ax8, 0, nothing)

rowsize!(f.layout, 2, Relative(2/3))

display(f)
# save(joinpath(fig_dir, "fig2_network_training_and_param_selection.pdf"), f)




##------------- KS test for whether ascending vs descending from same distribution


dt = @pipe df |> 
@transform(_, :mi = ifelse.(:mi .< 0, 0, :mi)) |> 
@subset(_, :peak_mi, :muscle .== "all", :nspikes .> 1000)

ApproximateTwoSampleKSTest(dt[dt.direction .== "ascending", :mi], dt[dt.direction .== "descending", :mi])

ApproximateTwoSampleKSTest(dt[dt.direction .== "ascending", :precision], dt[dt.direction .== "descending", :precision])

##

dt = @pipe df |> 
@transform(_, :mi = ifelse.(:mi .< 0, 0, :mi)) |> 
@subset(_, :peak_mi, :muscle .== "all", :nspikes .> 1000)

dt.fr = dt.nspikes ./ dt.flapping_time

f = Figure()
ax = Axis(f[1,1])
scatter!(ax, dt.mi, dt.fr)
ablines!(ax, [0], [1], color=:black)
f

f, ax, hs = hist(dt.mi ./ dt.fr)

eta = dt.mi ./ (dt.fr .* log2.(exp(1) ./ (dt.fr .* (dt.precision ./ 1000))))
eta = eta[eta .> 0]

##

dt = @pipe df |> 
    @subset(_, :muscle .== "all", :nspikes .> 1000) |> 
    @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi)) |> 
    groupby(_, [:moth, :neuron, :muscle]) |> 
    @transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), false, true)) |> 
    @transform(_, :window_select = ifelse.(:has_timing_info, :peak_valid_mi, :peak_mi)) |> 
    @subset(_, :window_select)

ApproximateTwoSampleKSTest(dt[dt.has_timing_info, :mi], dt[(!).(dt.has_timing_info), :mi])

## -------------------------------- Figure 4: Redundancy

dfr = DataFrame()
for task in vcat(0:11)
    read_redundancy_file!(dfr, joinpath(data_dir, "2025-07-25_main_redundancy_PACE_task_$(task).h5"), task)
end

# dfr = leftjoin(dfr, df_direction, on=[:moth, :neuron]) # Add direction?

# Get optimal window sizes
dfr = @pipe dfr |> 
@transform!(_, :mi = :mi ./ :window, :window = :window .* 1000) |> 
groupby(_, [:moth, :neuron, :muscle]) |> 
transform!(_, [:window, :mi, :precision] => get_optimal_mi_precision => [:max_valid_window, :peak_mi, :peak_valid_mi])

# Modify MI to be value across all flapping time (not conditional on activity)
function get_proportion_of_active_windows_redundancy(moth, neuron, window)
    thismoth = replace(moth[1], r"-1$" => "_1")
    spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
    bouts = npzread(joinpath(data_dir, "..", thismoth * "_bouts.npz"))
    neuron_split = split.(neuron, "-")
    # Get all unique single neurons and window sizes
    unique_neurons = unique(vcat(neuron_split...))
    unique_windows = unique(window)
    # Pre-compute window assignments for each unique neuron and window size
    # Structure: window_cache[neuron_name][window_size][bout_idx] = Set of unique window indices
    window_cache = Dict(
        n => Dict{Float64, Vector{Set{Int}}}() 
        for n in unique_neurons
    )
    for neur_name in unique_neurons
        for wind in unique_windows
            wind_samples = (wind / 1000 * 30000)
            unique_assignments = Vector{Set{Int}}(undef, length(bouts["starts"]))
            for i in eachindex(bouts["starts"])
                mask = (spikes[neur_name] .>= bouts["starts"][i]) .& 
                       (spikes[neur_name] .< bouts["ends"][i])
                spikes_in_bout = spikes[neur_name][mask]
                if length(spikes_in_bout) == 0
                    unique_assignments[i] = Set{Int}()
                else
                    windows = collect(bouts["starts"][i]:wind_samples:bouts["ends"][i])
                    window_assignments = searchsortedlast.(Ref(windows), spikes_in_bout)
                    unique_assignments[i] = Set(window_assignments)
                end
            end
            window_cache[neur_name][wind] = unique_assignments
        end
    end
    
    # Now compute fraction active for each neuron pair using cached data
    frac_active_dict = Dict(n => Dict{Float64, Float64}() for n in neuron_split)
    for neur_strings in neuron_split
        for wind in unique_windows
            wind_samples = (wind / 1000 * 30000)
            frac_active_bouts = zeros(length(bouts["starts"]))
            for i in eachindex(bouts["starts"])
                # Access pre-computed unique window assignments
                unique_windows_1 = window_cache[neur_strings[1]][wind][i]
                unique_windows_2 = window_cache[neur_strings[2]][wind][i]
                if isempty(unique_windows_1) && isempty(unique_windows_2)
                    frac_active_bouts[i] = 0.0
                else
                    # Union of the two sets gives all unique windows with spikes
                    full_windows = length(union(unique_windows_1, unique_windows_2))
                    windows = collect(bouts["starts"][i]:wind_samples:bouts["ends"][i])
                    total_windows = length(windows)
                    frac_active_bouts[i] = full_windows / total_windows
                end
            end
            frac_active_dict[neur_strings][wind] = mean(frac_active_bouts)
        end
    end
    frac_active_vec = [frac_active_dict[n][w] for (n,w) in zip(neuron_split, window)]
    return frac_active_vec
end
# Save MI values converted to bits/s of flapping time, rather than bits/s of any activity
dfr = @pipe dfr |> 
groupby(_, [:moth]) |> 
transform!(_, [:moth, :neuron, :window] => get_proportion_of_active_windows_redundancy => :frac_active) |> 
@transform!(_, :mi = :mi .* :frac_active)

# Set MI of neuron pairs where both had fewer than 1000 spikes to NaN
nspike_dict = Dict(tuple(row.moth, row.neuron) => row.nspikes for row in eachrow(df_neuronstats))
dfr = @pipe dfr |> 
DataFrames.transform(_, [:moth, :neuron] => ByRow() do m,n
    n1, n2 = parse.(Int64, split(n, "-"))
    nspike1, nspike2 = nspike_dict[m,n1], nspike_dict[m,n2]
    (; nspike1, nspike2)
end => AsTable) |> 
@transform(_, :mi = ifelse.((:nspike1 .>= 1000) .|| (:nspike2 .>= 1000), :mi, NaN))


##
using LinearAlgebra

function fig_redundancy()
inch = 96
cm = inch / 2.54
f = Figure(size=(800*1.56, 500))
# f = Figure(size=(17.5cm, 0.4*17.5cm))
ga = f[1,1] = GridLayout()
ax = [Axis(ga[i,j], aspect=DataAspect()) for i in 1:2, j in 1:3]

newmoths = [replace(m, r"_1$" => "-1") for m in moths]
matlist = []
good_neuron_dict = Dict{String, Any}()
for (ii, (axi,axj,moth)) in enumerate(Iterators.zip(repeat(1:2, inner=3), repeat(1:3,2), newmoths))
    sdf = @pipe df |> 
        @subset(_, :peak_mi, :muscle .== "all", :moth .== moth) |> 
        @transform(_, :mi = ifelse.(:mi .> 0, :mi, 0))
    dfrm = @pipe dfr |> 
        @subset(_, :moth .== sdf.moth[1], :peak_mi) |> 
        @transform(_, :mi = ifelse.(:mi .> 0, :mi, 0))
    # Arrange neurons by firing rate
    sort!(sdf, [order(:label), order(:mi)])
    neurons = unique(sdf.neuron)
    # Populate matrix
    mat = zeros(length(neurons), length(neurons))
    for row in eachrow(dfrm)
        npair = parse.(Float64, split(row.neuron, "-"))
        i = findfirst(neurons .== npair[1])
        j = findfirst(neurons .== npair[2])
        i_mi = @subset(sdf, :neuron .== npair[1]).mi
        j_mi = @subset(sdf, :neuron .== npair[2]).mi
        if length(i_mi) > 0 && length(j_mi) > 0
            mat[i,j] = row.mi - (i_mi[1] + j_mi[1])
            mat[j,i] = row.mi - (i_mi[1] + j_mi[1])
        end
    end
    push!(matlist, mat)

    # Update axis
    ax[axi,axj].title = "Moth $(ii)" #moth
    good_neuron_dict[moth] = [sum(sdf.label .== "good"), nrow(sdf)]
end


colrange = [-maximum(maximum(x) for x in matlist), maximum(maximum(x) for x in matlist)]
for (axi,axj,moth,mat) in Iterators.zip(repeat(1:2, inner=3), repeat(1:3, 2), newmoths, matlist)
    heatmap!(ax[axi,axj], mat,
        colormap=:seismic, colorrange=colrange
    )
    bracket!(ax[axi,axj], good_neuron_dict[moth][1]+1, 0, good_neuron_dict[moth][2], 0, 
        text = "MUA",
        orientation = :up,  # Bracket opens downward
        textoffset = 10,      # Distance of text from bracket
        fontsize = 12
    )
    bracket!(ax[axi,axj], 0, good_neuron_dict[moth][1]+1, 0, good_neuron_dict[moth][2], 
        text = "MUA",
        orientation = :down,  # Bracket opens downward
        textoffset = 10,      # Distance of text from bracket
        fontsize = 12
    )
    ax[axi,axj].xticks = ([1, good_neuron_dict[moth]...], ["1", string(good_neuron_dict[moth][1]), string(good_neuron_dict[moth][2])])
    ax[axi,axj].yticks = ([1, good_neuron_dict[moth]...], ["1", string(good_neuron_dict[moth][1]), string(good_neuron_dict[moth][2])])
end
Colorbar(ga[:, end+1], colormap=:seismic, colorrange=colrange, label="II (bits/s)")

apply_letter_label(ga, "A")

# Get all precision and II values as vectors
prec_change, ii, ii_prec = Float64[], Float64[], Float64[]
for moth in newmoths
    # II uses all neurons
    sdf = @subset(df, :peak_mi, :muscle .== "all", :moth .== moth)
    dfrm = @subset(dfr, :moth .== moth, :peak_mi)
    # Arrange neurons by firing rate
    sort!(sdf, [order(:label), order(:meanrate)])
    neurons = unique(sdf.neuron)
    # Populate matrix
    for row in eachrow(dfrm)
        npair = parse.(Float64, split(row.neuron, "-"))
        i_mi = @subset(sdf, :neuron .== npair[1]).mi
        j_mi = @subset(sdf, :neuron .== npair[2]).mi
        if length(i_mi) > 0 && length(j_mi) > 0
            push!(ii, row.mi - (i_mi[1] + j_mi[1]))
        end
    end
    # Precision has to use only neurons with timing info
    sdf = @subset(df, :peak_valid_mi, :muscle .== "all", :moth .== moth)
    dfrm = @subset(dfr, :moth .== moth, :peak_valid_mi)
    # Arrange neurons by firing rate
    sort!(sdf, [order(:label), order(:meanrate)])
    neurons = unique(sdf.neuron)
    # Populate matrix
    for row in eachrow(dfrm)
        npair = parse.(Float64, split(row.neuron, "-"))
        i_prec = @subset(sdf, :neuron .== npair[1]).precision
        j_prec = @subset(sdf, :neuron .== npair[2]).precision
        i_mi = @subset(sdf, :neuron .== npair[1]).mi
        j_mi = @subset(sdf, :neuron .== npair[2]).mi
        if length(i_prec) > 0 && length(j_prec) > 0
            # append!(prec_change, row.precision - mean([i_prec[1], j_prec[1]]))
            append!(prec_change, row.precision - min(i_prec[1], j_prec[1]))
            push!(ii_prec, row.mi - (i_mi[1] + j_mi[1]))
        end
    end
end

# bins = range(minimum(prec_change), maximum(prec_change), 30)

# resample_cmap(:lisbon, length(bins))
# hist!(axh, prec_change, normalization=:pdf, bins=bins)

g2 = f[1,2] = GridLayout()
gb = g2[1,1] = GridLayout()
axh = Axis(gb[1,1], xlabel="II (bits/s)", ylabel="Count")
poly!(axh, Point2f[(0,0), (1000, 0), (0, 1000)],
    color=resample_cmap(:seismic, 10)[end-2], alpha=0.2
)
poly!(axh, Point2f[(0,0), (-1000, 0), (0, 1000)],
    color=resample_cmap(:seismic, 10)[3], alpha=0.2
)
min_ii = minimum(x->isnan(x) ? 0 : x,ii)
max_ii = maximum(x->isnan(x) ? 0 : x,ii)
bins = range(-max_ii, max_ii, 201)
hist!(axh, ii[ii .< 0], bins=bins, color=resample_cmap(:seismic, 10)[2])
hist!(axh, ii[ii .> 0], bins=bins, color=resample_cmap(:seismic, 10)[end-1])
vlines!(axh, 0, color=:black)
text!(axh, 0.15, 0.5, text="Redundant", 
    fontsize=18,
    color=resample_cmap(:seismic, 10)[2],
    align=(:center, :bottom),
    space=:relative
)
text!(axh, 0.85, 0.5, text="Synergistic", 
    fontsize=18,
    color=resample_cmap(:seismic, 10)[end-1], 
    space=:relative, 
    align=(:center, :bottom)
)
xlims!(axh, -10, 25)
ylims!(axh, 0, 200)
apply_letter_label(gb, "B")

gc = g2[2,1] = GridLayout()
axp = Axis(gc[1,1], 
    xlabel="(Paired precision) - (Mean single precision), (ms)",
    # xlabel=L"$\tau(X_i,X_j;Y) - \frac{\tau(X_i;Y) + \tau(X_j;Y)}{2}$ (ms)",
    ylabel="Prob. density")
hist!(axp, prec_change, bins=100, normalization=:pdf)
text!(axp, 0.15, 0.5, text="Single more \n precise", fontsize=16, space=:relative, align=(:center, :bottom))
text!(axp, 0.85, 0.5, text="Paired more \n precise", fontsize=16, space=:relative, align=(:center, :bottom))
vlines!(axp, 0, color=:black)
ylims!(axp, 0, nothing)
apply_letter_label(gc, "C")

rowgap!(g2, 0)

colsize!(f.layout, 1, Relative(0.6))
return f
end

f = fig_redundancy()
save(joinpath(fig_dir, "fig4_redundancy.pdf"), f)
f

##  Plot amount of new information against amount of new precision. II with "interaction precision"
# Right now these differ on how many II calculations there are.... Double check this nonsense. THIS NONSENSE
dfrm = @subset(dfr, :peak_valid_mi)
dfrm.ii .= NaN
dfrm.ip .= NaN
dfrm.direction .= "UU"
sdf = @subset(df, :peak_valid_mi, :muscle .== "all")
# Populate matrix
for (i,row) in enumerate(eachrow(dfrm))
    npair = parse.(Float64, split(row.neuron, "-"))
    idf = @subset(sdf, :neuron .== npair[1], :moth .== row.moth)
    jdf = @subset(sdf, :neuron .== npair[2], :moth .== row.moth)
    i_mi, j_mi = idf.mi, jdf.mi
    i_prec, j_prec = idf.precision, jdf.precision
    # if length(i_mi) > 0 && length(j_mi) > 0
    if length(i_prec) > 0 && length(j_prec) > 0
        dfrm.ii[i] = row.mi - (i_mi[1] + j_mi[1])
        dfrm.ip[i] = row.precision - min(i_prec[1], j_prec[1])
        directs = sort([uppercase(idf.direction[1][1]), uppercase(jdf.direction[1][1])])
        dfrm.direction[i] = directs[1] * "-" * directs[2]
    end
end

dfrm = @subset(dfrm, (!).(isnan.(:ip)) .&& (!).(isnan.(:ii)))

f = Figure()

axtop = Axis(f[1,1])
axcenter = Axis(f[2,1],
    xlabel="I(Xᵢ,Xⱼ; Y) - (I(Xᵢ;Y) + I(Xⱼ;Y))  (bits/s)",
    ylabel="τ(Xᵢ,Xⱼ; Y) - min(τ(Xᵢ;Y), τ(Xⱼ;Y))  (ms)"
)
axright = Axis(f[2,2])

# Shading
poly!(axcenter, Point2f[(0,-1000), (1000, 0), (0, 1000)],
    color=resample_cmap(:seismic, 10)[end-2], alpha=0.2,
    xautolimits=false, yautolimits=false)
poly!(axcenter, Point2f[(0,-1000), (-1000, 0), (0, 1000)],
    color=resample_cmap(:seismic, 10)[3], alpha=0.2,
    xautolimits=false, yautolimits=false)
poly!(axtop, Point2f[(0,-1000), (1000, 0), (0, 1000)],
    color=resample_cmap(:seismic, 10)[end-2], alpha=0.2,
    xautolimits=false, yautolimits=false)
poly!(axtop, Point2f[(0,-1000), (-1000, 0), (0, 1000)],
    color=resample_cmap(:seismic, 10)[3], alpha=0.2,
    xautolimits=false, yautolimits=false)

text!(axtop, 0.15, 0.5, text="Redundant", 
    fontsize=18, color=resample_cmap(:seismic, 10)[2],
    align=(:center, :bottom), space=:relative)
text!(axtop, 0.85, 0.5, text="Synergistic", 
    fontsize=18, color=resample_cmap(:seismic, 10)[end-1],
    align=(:center, :bottom), space=:relative)
text!(axright, 0.5, 0.9, text="Paired \n less precise", 
    fontsize=18, color=:black,
    align=(:center, :center), space=:relative)
text!(axright, 0.5, 0.1, text="Paired \n more precise", 
    fontsize=18, color=:black,
    align=(:center, :center), space=:relative)


maxval_ii = max(abs(minimum(dfrm.ii)), maximum(dfrm.ii))
maxval_ip = max(abs(minimum(dfrm.ip)), maximum(dfrm.ip))
ii_bins = range(-maxval_ii, maxval_ii, 81)
ip_bins = range(-maxval_ii, maxval_ii, 41)
hist!(axtop, dfrm.ii[dfrm.ii .< 0], bins=ii_bins, color=resample_cmap(:seismic, 10)[2])
hist!(axtop, dfrm.ii[dfrm.ii .> 0], bins=ii_bins, color=resample_cmap(:seismic, 10)[end-1])
scatter!(axcenter, dfrm.ii, dfrm.ip, color=RGB(0.25), markersize=6)
hist!(axright, dfrm.ip, direction=:x, bins=ip_bins, color=RGB(0.35))

vlines!(axtop, 0, color=:black, linewidth=2)
hlines!(axright, 0, color=:black, linewidth=2)
hlines!(axcenter, 0, color=:black, linewidth=2)
vlines!(axcenter, 0, color=:black, linewidth=2)

linkxaxes!(axcenter, axtop)
linkyaxes!(axcenter, axright)
ylims!(axtop, low=0)
xlims!(axright, low=0)
xl = (-maxval_ii-0.5, maxval_ii+0.5)
yl = (-maxval_ip-0.5, maxval_ip+0.5)
xlims!(axcenter, xl...)
ylims!(axcenter, yl...)

hidedecorations!(axtop)
hidedecorations!(axright)
rowsize!(f.layout, 1, Relative(1/4))
colsize!(f.layout, 2, Relative(1/4))
rowgap!(f.layout, 0)
colgap!(f.layout, 0)

display(f)
##

# II uses all neurons
dfrm = @subset(dfr, :peak_valid_mi)
dfrm.ii .= 0.0
dfrm.pi .= 0.0
dfrm.direction .= "UU"
sdf = @subset(df, :peak_valid_mi, :muscle .== "all")
# Populate matrix
for (i,row) in enumerate(eachrow(dfrm))
    npair = parse.(Float64, split(row.neuron, "-"))
    idf = @subset(sdf, :neuron .== npair[1], :moth .== row.moth)
    jdf = @subset(sdf, :neuron .== npair[2], :moth .== row.moth)
    i_mi, j_mi = idf.mi, jdf.mi
    i_prec, j_prec = idf.precision, jdf.precision
    if length(i_mi) > 0 && length(j_mi) > 0
        dfrm.ii[i] = row.mi - (i_mi[1] + j_mi[1])
        dfrm.pi[i] = row.precision - min(i_prec[1], j_prec[1])
        directs = sort([uppercase(idf.direction[1][1]), uppercase(jdf.direction[1][1])])
        dfrm.direction[i] = directs[1] * "-" * directs[2]
    end
end


CairoMakie.activate!()

plt = (
mapping([0], [0]) * visual(ABLines, color=:black) + 
mapping([0]) * visual(VLines, color=:black) +
AlgebraOfGraphics.data(@subset(dfrm, :ii .!= 0)) * 
mapping(:ii=>L"II(X_1, X_2; Y) \text{ (bits/s)}", :pi=>L"\tau (X_1,X_2;Y) - min(\tau(X_1;Y),\text{ } \tau(X_2;Y)) \text{ (ms)}", 
    color=:direction=>"Direction", row=:direction) * 
visual(Scatter, alpha=0.7)
)
f = draw(plt, figure=(; size=(400, 800)))
display(f)

save(joinpath(fig_dir, "supp_precision_interactioninfo.pdf"), f)
## -------------------------------- Redundancy: Individual information vs II
newmoths = [replace(m, r"_1$" => "-1") for m in moths]
for (ii, (axi,axj,moth)) in enumerate(Iterators.zip(repeat(1:2, inner=3), repeat(1:3,2), newmoths))
    sdf = @pipe df |> 
        @subset(_, :peak_mi, :muscle .== "all", :moth .== moth) |> 
        @transform(_, :mi = ifelse.(:mi .> 0, :mi, 0))
    dfrm = @pipe dfr |> 
        @subset(_, :moth .== sdf.moth[1], :peak_mi) |> 
        @transform(_, :mi = ifelse.(:mi .> 0, :mi, 0))
    # Arrange neurons by firing rate
    sort!(sdf, [order(:label), order(:mi)])
    neurons = unique(sdf.neuron)
    # Populate matrix
    mat = zeros(length(neurons), length(neurons))
    for row in eachrow(dfrm)
        npair = parse.(Float64, split(row.neuron, "-"))
        i = findfirst(neurons .== npair[1])
        j = findfirst(neurons .== npair[2])
        i_mi = @subset(sdf, :neuron .== npair[1]).mi
        j_mi = @subset(sdf, :neuron .== npair[2]).mi
        if (length(i_mi) == 0) || (length(j_mi) == 0)
            println("$(moth) $(npair) has failed")
        end
        if length(i_mi) > 0 && length(j_mi) > 0
            mat[i,j] = row.mi - (i_mi[1] + j_mi[1])
            mat[j,i] = row.mi - (i_mi[1] + j_mi[1])
        end
    end
end


##
dt = @pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all") |> 
# @subset(_, :mi .> 0, :peak_mi) |> 
leftjoin(_, dfc, on=[:moth, :neuron]) |> 
@subset(_, :omnibus_stat .!= 0) |> 
@transform(_, :peak_power = 10 .* log10.(:peak_power))

f = Figure()
ax = Axis(f[1,1])

for (i,gdf) in enumerate(groupby(dt, :direction))
    scatter!(ax, gdf.peak_power, rand(nrow(gdf)) .* 0.2 .+ i)
end
f

## ---------------- Look at phasic properties of neurons against motor information
@pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all") |> 
# @subset(_, :mi .> 0, :peak_mi) |> 
leftjoin(_, dfc, on=[:moth, :neuron]) |> 
@subset(_, :omnibus_stat .!= 0) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:kuiper_stat, :mi, color=:has_timing_info, layout=:muscle) * (visual(Scatter))# + linear())
) |> 
draw(_, axis=(; xscale=log10, yscale=log10))

## Summary statistics of different circular stats
mi_cor = @pipe df |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all") |> 
leftjoin(_, dfc, on=[:moth, :neuron]) |> 
@subset(_, :omnibus_stat .!= 0) |> 
combine(_, 
    [[:kuiper_stat, :mi], [:omnibus_stat, :mi], [:watson_stat, :mi], [:rao_stat, :mi], [:peak_power, :mi], [:total_power, :mi]] 
    .=> ((a,b) -> cor(a,b)),
    [[:kuiper_stat, :mi], [:omnibus_stat, :mi], [:watson_stat, :mi], [:rao_stat, :mi], [:peak_power, :mi], [:total_power, :mi]] .=> ((a,b) -> cor(log.(a),log.(b))) 
    .=> [:log_kuiper, :log_omnibus, :log_watson, :log_rao, :log_peak, :log_total]) |> 
println(_)
mi_prec = @pipe df |> 
@subset(_, :mi .> 0, :peak_valid_mi, :muscle .== "all") |> 
leftjoin(_, dfc, on=[:moth, :neuron]) |> 
@subset(_, :omnibus_stat .!= 0) |> 
combine(_, 
    [[:kuiper_stat, :precision], [:omnibus_stat, :precision], [:watson_stat, :precision], [:rao_stat, :precision], [:peak_power, :precision], [:total_power, :precision]] 
    .=> ((a,b) -> cor(a,b)),
    [[:kuiper_stat, :precision], [:omnibus_stat, :precision], [:watson_stat, :precision], [:rao_stat, :precision], [:peak_power, :precision], [:total_power, :precision]] .=> ((a,b) -> cor(log.(a),log.(b))) 
    .=> [:log_kuiper, :log_omnibus, :log_watson, :log_rao, :log_peak, :log_total]) |> 
println(_)


##

f = Figure()
ax = Axis(f[1,1])
thismoth = replace.(moths[1], r"-1$" => "_1")
spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
vlines!(ax, spikes["ldlm"] ./ fsamp, ymin=0.0, ymax=0.5)
vlines!(ax, spikes["27"] ./ fsamp, ymin=0.5, ymax=1.0)
f

##
f = Figure()

spec_range = function(neur, spikes; wingbeat_freq_range=[1,200], fsamp=fsamp)
    discrete_vec = zeros(spikes[neur][end]+10)
    discrete_vec[spikes[neur]] .= 1
    pxx = welch_pgram(discrete_vec, fsamp * 20; fs=fsamp)

    fi = findfirst(pxx.freq .> wingbeat_freq_range[1])
    li = findlast(pxx.freq .< wingbeat_freq_range[2])
    return pxx.freq[fi:li], pxx.power[fi:li]
end

for (i,moth) in enumerate(moths)
    spikes = npzread(joinpath(data_dir, "..", moth * "_data.npz"))
    ax = Axis(f[i,1])
    pfreq, ppower = spec_range("ldlm", spikes; wingbeat_freq_range=wingbeat_freq_range)
    lines!(ax, pfreq, ppower)
    xlims!(ax, 0, 50)
end
f

##

@pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all") |> 
leftjoin(_, dfc, on=[:moth, :neuron]) |> 
(
AlgebraOfGraphics.data(_) * mapping(:peak_power, :mi, color=:direction) * visual(Scatter)
) |> 
draw(_, axis=(; xscale=log10, yscale=log10))