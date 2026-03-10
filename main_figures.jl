using HDF5
using CSV
using NPZ
using DelimitedFiles
using Arrow
using FileIO
using Statistics
using StatsBase # Mainly for rle
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
for i in 1:3
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


##------------- Likelihood ratio test for pareto vs exponential distribution in neuron information

dt = @pipe df |> 
    @subset(_, :muscle .== "all", :nspikes .> 1000) |> 
    @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi)) |> 
    groupby(_, [:moth, :neuron, :muscle]) |> 
    # transform(_, [:peak_valid_mi]) |> 
    @transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
    @transform(_, :window_select = ifelse.(:peak_off_valid .== "Spike timing info", :peak_valid_mi, :peak_mi)) |> 
    @subset(_, :window_select)
dt = dt[sortperm(dt.peak_off_valid), :]

f = Figure()


##------------- KS test for whether ascending vs descending from same distribution

using HypothesisTests

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

##
using LinearAlgebra
using MultivariateStats
using Arpack

f = Figure(size=(800*1.56, 500))
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
prec_change, ii = Float64[], Float64[]
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
        if length(i_prec) > 0 && length(j_prec) > 0
            append!(prec_change, row.precision - mean([i_prec[1], j_prec[1]]))
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
bins = range(minimum(ii), maximum(ii), 101)
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
xlims!(axh, -25, 25)
ylims!(axh, 0, 400)
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

# save(joinpath(fig_dir, "fig4_redundancy.pdf"), f)
f
# Could do this as hist with scatter of II and precision: One day!

##

# II uses all neurons
dfrm = @subset(dfr, :peak_valid_mi)
dfrm.ii .= 0.0
dfrm.direction .= "UU"
sdf = @subset(df, :peak_valid_mi, :muscle .== "all")
# Arrange neurons by firing rate
neurons = unique(sdf.neuron)
# Populate matrix
for (i,row) in enumerate(eachrow(dfrm))
    npair = parse.(Float64, split(row.neuron, "-"))
    idf = @subset(sdf, :neuron .== npair[1], :moth .== row.moth)
    jdf = @subset(sdf, :neuron .== npair[2], :moth .== row.moth)
    i_mi, j_mi = idf.mi, jdf.mi
    if length(i_mi) > 0 && length(j_mi) > 0
        dfrm.ii[i] = row.mi - (i_mi[1] + j_mi[1])
        directs = sort([uppercase(idf.direction[1][1]), uppercase(jdf.direction[1][1])])
        dfrm.direction[i] = directs[1] * "-" * directs[2]
    end
end

# f = Figure()
# ax = Axis(f[1,1])
# mask = dfrm.ii .!= 0
# scatter!(ax, dfrm.ii[mask], dfrm.precision[mask])
# f
@pipe dfrm |> 
@subset(_, :ii .!= 0) |> 
(AlgebraOfGraphics.data(_) * 
mapping(:ii, :precision, color=:direction) * visual(Scatter, alpha=0.7)
) |> 
draw(_)

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

df_neurons = @pipe df |> 
@subset(_, :peak_mi, :muscle .== "all") |> 
@transform(_, :n1 = :neuron, :n2 = :neuron)

@pipe dfr |>
@subset(_, :peak_mi) |> 
@transform(_, 
    :n1 = parse.(Float64, getindex.(split.(:neuron, "-"), 1)), 
    :n2 = parse.(Float64, getindex.(split.(:neuron, "-"), 2))) |> 
leftjoin(_, select(df_neurons, :moth, :n1, :mi, :precision), on=[:moth, :n1], renamecols=""=>"_n1") |> 
leftjoin(_, select(df_neurons, :moth, :n2, :mi, :precision), on=[:moth, :n2], renamecols=""=>"_n2") |> 
@transform(_, :ii = :mi .- (:mi_n1 .+ :mi_n2)) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:mi_n1, :mi_n2, color=:ii, row=:moth) * visual(Scatter)
) |> 
draw(_)

## -------------------------------- Redundancy: Clique percolation to find subpopulations
using GraphMakie
using GraphMakie.NetworkLayout
using ColorSchemes

colrange = [-maximum(maximum(x) for x in matlist), maximum(maximum(x) for x in matlist)]


f = Figure()
for (mi, moth) in enumerate(newmoths)
    thismat = matlist[mi]
    # clique = clique_percolation(SimpleGraph(thismat .> (0.5 * maximum(thismat))), k=2)
    g = SimpleGraph(thismat)
    inds1 = [src(edge) for edge in edges(g)]
    inds2 = [dst(edge) for edge in edges(g)]
    weights = [thismat[i1,i2] for (i1,i2) in Iterators.zip(inds1, inds2)]

    axr = Axis(f[1,mi], title="Redundant unit pairs")
    axs = Axis(f[2,mi], title="Synergistic unit pairs")

    # Redundant pairs graph
    thresh = percentile(weights[weights .< 0], 25)
    # g = SimpleGraph(thismat .< (0.5 * minimum(thismat)))
    g = SimpleGraph(thismat .< thresh)
    cliques = maximal_cliques(g)
    inds = [(src(edge), dst(edge)) for edge in edges(g)]
    values = [(thismat[i[2], i[1]] - colrange[1]) / (colrange[2] - colrange[1])  for i in inds]
    colors = get.(Ref(colorschemes[:seismic]), values)
    graphplot!(axr, g, layout=Spring(), edge_color=colors)
    # Synergistic pairs graph
    thresh = percentile(weights[weights .> 0], 75)
    # g = SimpleGraph(thismat .> (0.5 * maximum(thismat)))
    g = SimpleGraph(thismat .> thresh)
    cliques = maximal_cliques(g)
    inds = [(src(edge), dst(edge)) for edge in edges(g)]
    values = [(thismat[i[2], i[1]] - colrange[1]) / (colrange[2] - colrange[1])  for i in inds]
    colors = get.(Ref(colorschemes[:seismic]), values)
    graphplot!(axs, g, layout=Spring(), edge_color=colors)

    hidedecorations!(axr)
    hidedecorations!(axs)
end
f

## -------------------------------- Redundancy: Look at window size plots for both neurons in a pair, and the final pair
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


dfr_row = rand(eachrow(dfr))
moth = dfr_row.moth
neuron_1, neuron_2 = parse.(Float64, split(dfr_row.neuron, "-"))

f = Figure(size=(1000,500))

dt = @subset(dfr, :neuron .== dfr_row.neuron, :moth .== dfr_row.moth)
ax11, ax12 = plot_mi_precision_against_window!(f, (1,1), dt[sortperm(dt.window),:])
dt = @subset(df, :neuron .== neuron_1, :moth .== dfr_row.moth, :muscle .== "all")
ax21, ax22 = plot_mi_precision_against_window!(f, (1,2), dt[sortperm(dt.window),:])
dt = @subset(df, :neuron .== neuron_2, :moth .== dfr_row.moth, :muscle .== "all")
ax31, ax32 = plot_mi_precision_against_window!(f, (1,3), dt[sortperm(dt.window),:])

f
## ---------------------------------------- I(Neurons; kinematics)

## ---------------------------------------- Phasic neurons explain information figure (Figure 5?) 
muscle_colors = [
    "lax" => "#94D63C", "rax" => "#6A992A",
    "lba" => "#AE3FC3", "rba" => "#7D2D8C",
    "lsa" => "#FFBE24", "rsa" => "#E7AC1E",
    "ldvm"=> "#66AFE6", "rdvm"=> "#2A4A78",
    #"ldlm"=> "#E87D7A", "rdlm"=> "#C14434"
]
color_dict = Dict(
    "descending" => Makie.wong_colors()[1], 
    "ascending" => Makie.wong_colors()[2], 
    "uncertain" => Makie.wong_colors()[3])
timing_colors = Dict("No spike timing info" => "#5e3c99", "Spike timing info" => "#e66101")
example_neurons = ["76", "27", "11", "88"] # Descending order of mi, but will be plot in reverse order (makes color/draw order work)
example_neuron_colors = Dict(
    "76" => Makie.wong_colors()[1],
    "27" => Makie.wong_colors()[5],
    "11" => Makie.wong_colors()[4],
    "88" => Makie.wong_colors()[7]
)

# function phasic_neuron_figure()
f = Figure(size=(1900, 1500))

top_row = f[1,1:3] = GridLayout()

# -------- First panel, example neuron firing
ga = top_row[1, 1] = GridLayout()
ax_examp_range = [776, 778] .- 0.1
ax_examp = Axis(ga[1,1], 
    xticks=(ax_examp_range, ["0","2"]), limits=(tuple(ax_examp_range...), nothing),
    xticklabelspace = 0.0, xlabel="Time (s)"
)
wb_duration_thresholds = (32.5, 100) # ms ldlm
thismoth = replace.(moths[1], r"-1$" => "_1")
spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))

mask_neur = (spikes["76"] .> (ax_examp_range[1] - 1) * fsamp) .&& (spikes["76"] .< (ax_examp_range[2] + 1) * fsamp)
mask = (spikes["ldlm"] .> (ax_examp_range[1] - 1) * fsamp) .&& (spikes["ldlm"] .< (ax_examp_range[2] + 1) * fsamp)
spikevec = spikes["ldlm"][mask]

diffvec = diff(spikevec) ./ fsamp .* 1000 # units of ms
mask = (diffvec .> wb_duration_thresholds[1]) .&& (diffvec .< wb_duration_thresholds[2])
start_inds = spikevec[findall(vcat(false, mask))]
vspan!(ax_examp, start_inds[1:2:end-1] ./ fsamp, start_inds[2:2:end] ./ fsamp, color=RGBAf(0.9, 0.9, 0.9, 1.0))
vlines!(ax_examp, spikevec ./ fsamp, ymin=0, ymax=0.5, color=:black)
vlines!(ax_examp, spikes["76"][mask_neur] ./ fsamp, ymin=0.5, ymax=1.0)
textlabel!(ax_examp, 0.95, 0.25, text="LDLM", space=:relative, fontsize=18, text_align=(:right, :center))
textlabel!(ax_examp, 0.95, 0.75, text="Neuron 76", space=:relative, fontsize=18, text_align=(:right, :center), text_color=Makie.wong_colors()[1])
hideydecorations!(ax_examp)
hidespines!(ax_examp)
apply_letter_label(ga, "A")


# -------- Second panel, circular hist with muscles and a neuron
max_mi_neuron = @pipe df |> 
    @subset(_, :peak_mi, :nspikes .> 1000, :muscle .== "all", :label .== "good") |> 
    @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi), :moth = replace.(:moth, r"-1$" => "_1")) |> 
    groupby(_, :moth) |> 
    combine(_, [:neuron, :mi] => (n,mi) -> n[findmax(mi)[2]]) |> 
    @transform(_, :neuron_mi_function = string.(round.(Int, :neuron_mi_function)))
max_mi_neuron = Dict(Pair.(max_mi_neuron.moth, max_mi_neuron.neuron_mi_function))

gb = top_row[2,1] = GridLayout(padding=(0,0,0,0), tellwidth=false, tellheight=false)
phase_dict, wblen_dict, muscle_phase_dict = get_phase_dict(thismoth)
ax_muscle_hist = PolarAxis(gb[1:end,1], 
    radius_at_origin=-1,
    clip=false, clip_r=false
)
muscle_theta_r = Dict(
    "ax" => [pi+0.17*pi, 0.87], # Angle (radians) and radius of each text placement
    "ba" => [pi*0.3, 0.7],
    "sa" => [pi*0.08, 0.8],
    "dvm"=> [2.75, 1.1]
)
rlims!(ax_muscle_hist, -1.0, 0.6)
hiderdecorations!(ax_muscle_hist)
hidespines!(ax_muscle_hist)
lines!(ax_muscle_hist, [0, 0], [-1, 0.6], color=:black, linewidth=3)
for i in eachindex(muscle_colors)
    muscle = muscle_colors[i][1]
    if !(muscle in keys(muscle_phase_dict))
        continue
    end
    hist!(ax_muscle_hist, muscle_phase_dict[muscle] .* 2*pi, 
        bins=90, normalization=:pdf, #scale_to=1.0,
        color=(muscle_colors[i][2], 0.5)
    )
    # Muscle label for one muscle of each pair
    if muscle[1] == 'r'
        # mean_angle = mean(muscle_phase_dict[muscle] .* 2*pi)
        text!(ax_muscle_hist, muscle_theta_r[muscle[2:end]][1], muscle_theta_r[muscle[2:end]][2], 
            text=uppercase(muscle[2:end]), color=muscle_colors[i][2],
            fontsize=22
        )
    end
end
hist!(ax_muscle_hist, phase_dict[max_mi_neuron[moth]] .* 2*pi, 
    bins=ceil(Int, sqrt(length(phase_dict[max_mi_neuron[moth]]))), 
    normalization=:pdf, offset=-1, scale_to=1.0)
lines!(ax_muscle_hist, range(0, 2*pi, 1000), zeros(1000), color=:black, linewidth=2)
scatter!(ax_muscle_hist, 0, -1, color=:black, markersize=5)
text!(ax_muscle_hist, pi/2, -0.6, text="N. 76", font=:bold, color=:black, fontsize=16, align=(:center, :center))

lines!(ax_muscle_hist, [0, 0], [0.83, 0.98], color=:black, linewidth=3)
text!(ax_muscle_hist, 0, 1., text="LDLM", color=:black, fontsize=22, align=(:left, :center))

apply_letter_label(gb, "B")

# -------- Circular histograms of all "good" neurons in a moth
phase_dict, wblen_dict, muscle_phase_dict = get_phase_dict(thismoth)
gc = top_row[1:2, 2] = GridLayout()

# 1. Selection of good neurons (not MUA) from moth 1
phase_dict, wblen_dict, muscle_phase_dict = get_phase_dict(thismoth)
df_neuron = @pipe df |> 
    @subset(_, :moth .== thismoth, :label .== "good") |> 
    @subset(_, :peak_mi, :nspikes .> 1000, :muscle .== "all") |> 
    @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi))
sort!(df_neuron, [order(:label), order(:mi)])
# 2. For each neuron, draw polar histogram
polar_axes = []
polar_mis = Float64[]
for (i,row) in enumerate(eachrow(df_neuron))
    ri, ci = mod(i+1, 2) + 1, repeat(1:nrow(df_neuron), inner=2)[i]
    neur = string(round(Int, row.neuron))
    thisax = PolarAxis(top_row[1:2,2], 
        width=Relative(0.3), height=Relative(0.3), halign=-0.271 + 0.075*i, valign=0.075 + 0.85 * (ri-1),
        clip=false, clip_r=false
    )
    hiderdecorations!(thisax)
    hidethetadecorations!(thisax, grid=false, minorgrid=false)
    if neur in example_neurons
        hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[neur]))), 
            color=example_neuron_colors[neur])
        scatter!(thisax, 0, 0, color=:black, markersize=6)
    else
        hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[neur]))),
            color=:grey)
        scatter!(thisax, 0, 0, color=:black, markersize=6)
    end
    push!(polar_axes, thisax)
    push!(polar_mis, max(row.mi, 0.0))
end
# 3. Add the number line axis on the same grid cell
ax_nl = Axis(top_row[1:2,2], 
    width=Relative(1.0), height=Relative(1.0),
    halign=0.5, valign=0.5,
    backgroundcolor=:transparent
)
hidedecorations!(ax_nl)
hidespines!(ax_nl)
mi_min, mi_max = -0.1, maximum(polar_mis) * 1.1
y_min, y_max = 0.0, 1.0
xlims!(ax_nl, mi_min, mi_max)
ylims!(ax_nl, y_min, y_max)
line_pos = 0.5
lines!(ax_nl, [0, mi_max], [line_pos, line_pos], color=:black, linewidth=2) # Central number line
poly!(ax_nl, Point2f[(mi_max, line_pos),(mi_max-0.2, line_pos + 0.025), (mi_max-0.2, line_pos - 0.025)], color=:black)
for tick in [0.0, 10.0, 20.0]
    lines!(ax_nl, [tick, tick], line_pos .+ [-0.025, 0.025], color=:black, linewidth=2) # End tick
    textlabel!(ax_nl, tick, line_pos - 0.05, text=string(convert(Int, tick)), strokewidth=0)
end
text!(ax_nl, 0.7, 0.55, text="Mutual Information I(X;Y) (bits/s)", fontsize=22, align=(:left, :center), space=:relative)
# 4. Connect lines from polar hist plots to number line
on(events(f).window_open) do _  # fires once layout is resolved
    ax_nl_bb = ax_nl.layoutobservables.computedbbox[]
    ax_nl_origin = ax_nl_bb.origin
    ax_nl_widths = ax_nl_bb.widths
    for (pa, mi, i, neur) in zip(polar_axes, polar_mis, 1:length(polar_axes), string.(round.(Int, df_neuron.neuron)))
        pa_bb = pa.layoutobservables.computedbbox[]
        # Center of polar axis in scene pixels
        if mod(i,2) == 0 # Bottom row of polar axes
            px_center = pa_bb.origin .+ [pa_bb.widths[1], 0] ./ 2
        else
            px_center = pa_bb.origin .+ [pa_bb.widths[1] / 2, pa_bb.widths[2]]
        end
        # Convert scene pixels → ax_nl data coords
        x_frac = (px_center[1] - ax_nl_origin[1]) / ax_nl_widths[1]
        y_frac = (px_center[2] - ax_nl_origin[2]) / ax_nl_widths[2]
        pm = mod(i,2) == 0 ? 1 : -1
        x_data = mi_min + x_frac * (mi_max - mi_min)
        y_data = y_min + y_frac * y_max  + pm*0.01 # ylims are (0,1)
        # MI position on number line
        color = neur in example_neurons ? example_neuron_colors[neur] : :gray
        lines!(ax_nl, [x_data, mi], [y_data, line_pos], color=color, linewidth=0.8)
        scatter!(ax_nl, mi, line_pos, color=color, markersize=15)
        # Label position: centered on polar axis x, just above/below it
        px_x_center = pa_bb.origin[1] + pa_bb.widths[1] / 2
        x_label = mi_min + ((px_x_center - ax_nl_origin[1]) / ax_nl_widths[1]) * (mi_max - mi_min)
        # Label for neuron polar plot
        neur = string(round(Int, df_neuron[i, :neuron]))
        text_color = neur in example_neurons ? example_neuron_colors[neur] : :black
        if mod(i, 2) == 0  # top row: label above
            px_top = pa_bb.origin[2] + pa_bb.widths[2]
            y_frac = (px_top - ax_nl_origin[2]) / ax_nl_widths[2]
            y_label = y_min + y_frac * y_max
            text!(ax_nl, x_label, y_label, text="N. " * neur,
                align=(:center, :bottom), fontsize=14, color=text_color, font=:bold)
        else  # bottom row: label below
            px_bottom = pa_bb.origin[2]
            y_frac = (px_bottom - ax_nl_origin[2]) / ax_nl_widths[2]
            y_label = y_min + y_frac * y_max
            text!(ax_nl, x_label, y_label, text="N. " * neur,
                align=(:center, :top), fontsize=14, color=text_color, font=:bold)
        end
    end
end

apply_letter_label(gc, "C")
rowgap!(top_row, 0)
rowsize!(top_row, 1, Relative(1/4))
colsize!(top_row, 2, Relative(2/3))


bottom_row = f[2, 1:3] = GridLayout()
gd = bottom_row[1, 1] = GridLayout()
ge = bottom_row[1, 2] = GridLayout()
gf = bottom_row[1, 3] = GridLayout()


# -------- Power as predictor of MI, precision
wingbeat_freq_range = [1, 100]

spec_range = function(neur, spikes; wingbeat_freq_range=[1,200], fsamp=fsamp)
    discrete_vec = zeros(spikes[neur][end]+10)
    discrete_vec[spikes[neur]] .= 1
    pxx = welch_pgram(discrete_vec, fsamp * 10; fs=fsamp)

    fi = findfirst(pxx.freq .> wingbeat_freq_range[1])
    li = findlast(pxx.freq .< wingbeat_freq_range[2])
    return pxx.freq[fi:li], pxx.power[fi:li]
end

ax_freq = Axis(gd[1,1], xlabel="Frequency (Hz)", ylabel="Power spectral density (dB/Hz)")
vspan!(ax_freq, 17, 21, color=RGBf(0.8, 0.8, 0.8))
# Spectra, histogram for each neuron
power_vals = zeros(length(example_neurons))
for (i, neur) in enumerate(example_neurons)
    pfreq, ppower = spec_range(neur, spikes; wingbeat_freq_range=wingbeat_freq_range)
    mask = (pfreq .> 12) .&& (pfreq .< 24)
    ind = findmax(ppower[mask])[2] + findfirst(mask) - 1
    bump = neur .== "27" ? 1.0 : 0.0
    lines!(ax_freq, pfreq, 10 .* log10.(ppower) .+ bump, color=example_neuron_colors[neur])
    scatter!(ax_freq, pfreq[ind], 10 .* log10.(ppower[ind]) .+ bump, color=:black, markersize=15)
    scatter!(ax_freq, pfreq[ind], 10 .* log10.(ppower[ind]) .+ bump, color=example_neuron_colors[neur], markersize=10)
    power_vals[i] = 10 .* log10.(ppower[ind]) .+ bump
end
xlims!(ax_freq, 0, 50)

# Phase histograms ordered/sorted by peak power
ax_power_sort = Axis(gd[2,1])
hidedecorations!(ax_power_sort)
hidespines!(ax_power_sort)
power_polar_axes = []
for (i, examp_neuron) in enumerate(example_neurons[sortperm(power_vals)])
    this_inset = PolarAxis(gd[2,1], 
        width=Relative(0.4), height=Relative(0.4), halign=-0.5 + 0.4 * i, valign=0.5,
        clip=false, clip_r=false
    )
    hidedecorations!(this_inset)
    hist!(this_inset, phase_dict[examp_neuron] .* 2*pi, 
        normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[examp_neuron]))),
        color=example_neuron_colors[examp_neuron]
    )
    scatter!(this_inset, 0, 0, color=:black, markersize=5)
    push!(power_polar_axes, this_inset)
end
# Put "axis" line, labels for peak power
xmin, xmax = -80.5, -68
ymin, ymax = -0.25, 0.8
arrows2d!(ax_power_sort, Point2f(xmin, 0), [[xmax-xmin,0]])
ticks = Makie.get_tickvalues(LinearTicks(3), identity, xmin, xmax)
for lab_val in ticks
    lines!(ax_power_sort, [lab_val, lab_val], [-0.04, 0.04], color=:black)
    text!(ax_power_sort, lab_val, -0.06, text=string(lab_val), align=(:center, :top))
end
text!(ax_power_sort, 0.5, 0.05, text="Power at wingbeat frequency (dB/Hz)", 
    space=:relative, align=(:center, :center))
limits!(ax_power_sort, (xmin, xmax), (ymin, ymax))
let xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    on(events(f).window_open) do _  # fires once layout is resolved
        ax_p_bb = ax_power_sort.layoutobservables.computedbbox[]
        ax_p_origin = ax_p_bb.origin
        ax_p_widths = ax_p_bb.widths
        psort = sortperm(power_vals)
        for (pa, pval, neur) in zip(power_polar_axes, power_vals[psort], example_neurons[psort])
            pa_bb = pa.layoutobservables.computedbbox[]
            # Top of polar axis in scene pixels
            px_center = pa_bb.origin .+ [pa_bb.widths[1] / 2, pa_bb.widths[2]]
            # Convert scene pixels → ax_power_sort data coords
            x_data = xmin + ((px_center[1] - ax_p_origin[1]) / ax_p_widths[1]) * (xmax - xmin)
            y_data = ymin + ((px_center[2] - pa_bb.widths[2]/2 - ax_p_origin[2]) / ax_p_widths[2]) * ymax  # will require fine tuning
            # Power position on number line
            color = example_neuron_colors[neur]
            lines!(ax_power_sort, [x_data, pval], [y_data, 0], color=color, linewidth=0.8)
            sc = scatter!(ax_power_sort, pval, 0, color=color, markersize=15)
            translate!(sc, 0, 0, 10)
            # Label position: centered on polar axis x, just above/below it
            x_label = xmin + ((px_center[1] - ax_p_origin[1]) / ax_p_widths[1]) * (xmax - xmin)
            # Label above neuron polar plot
            px_top = pa_bb.origin[2] + pa_bb.widths[2]
            y_label = ymin + ((px_top - ax_p_origin[2]) / ax_p_widths[2]) * ymax + 0.15
            text!(ax_power_sort, x_label, y_label, text="N. " * neur,
                align=(:center, :bottom), fontsize=14, color=color, font=:bold)
        end
    end
end

df_power = @pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all") |> 
leftjoin(_, dfc, on=[:moth, :neuron])

gd_right = gd[1:2, 2] = GridLayout()
ax_power_mi = Axis(gd_right[1,1], yscale=log10, ylabel="I(X;Y) (bits/s)")
ax_power_prec = Axis(gd_right[2,1], yscale=log10, xlabel="Power at wingbeat frequency (dB/Hz)", ylabel="Spike timing precision (ms)")
linkxaxes!(ax_power_mi, ax_power_prec)
rowgap!(gd_right, 0)
colsize!(gd, 2, Relative(0.6))

for timing_info in ["Spike timing info", "No spike timing info"]
    gdf = @subset(df_power, :has_timing_info .== timing_info)
    scatter!(ax_power_mi, 10 .* log10.(gdf.peak_power), gdf.mi, 
        color=timing_colors[gdf.has_timing_info[1]],
        label=gdf.has_timing_info[1]
        # label=titlecase(first(gdf.direction)), 
        # color=color_dict[first(gdf.direction)]
    )
    if timing_info .== "Spike timing info"
        # mask = gdf.has_timing_info .== "Spike timing info"
        scatter!(ax_power_prec, 10 .* log10.(gdf.peak_power), gdf.precision,
            color=timing_colors[gdf.has_timing_info[1]])
            # color=color_dict[first(gdf.direction)])
    end
end
axislegend(ax_power_mi, position=:rb)
hidexdecorations!(ax_power_mi, grid=false, minorgrid=false)

apply_letter_label(gd, "D")

# -------- Circularity as predictor of MI, precision

df_circ = @pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all") |> 
leftjoin(_, dfc, on=[:moth, :neuron]) |> 
@subset(_, :omnibus_stat .!= 0)

ecdf_func = function(phase_spikes, x)
    sorted = sort(phase_spikes .* 2 * pi)
    return searchsortedlast.(Ref(sorted), x) ./ length(sorted)
end

ax_ecdf_examp = Axis(ge[:,1], 
    xlabel="Phase angle (Radians)", ylabel="Cumulative probability",
    xticks=([0, pi/2, pi, 3*pi/2, 2*pi], ["0", "π/2", "π", "3π/2", "2π"]),
    limits=((0,2*pi), (0,1))
)
ge_right = ge[1:2, 2] = GridLayout()
ax_circ_mi = Axis(ge_right[1,1], xscale=log10, yscale=log10, ylabel="I(X;Y) (bits/s)")
ax_circ_prec = Axis(ge_right[2,1], xscale=log10, yscale=log10, xlabel="Kuiper statistic V", ylabel="Spike timing precision (ms)")
linkxaxes!(ax_circ_mi, ax_circ_prec)
rowgap!(ge_right, 0)
colsize!(ge, 2, Relative(0.6))

unif_vec = range(0, 1, 1000)
for (i, examp_neuron) in enumerate(example_neurons)
    xvec = range(0, 2*pi, 1000)
    ecdf = ecdf_func(phase_dict[examp_neuron], xvec)
    comparison = ecdf .- unif_vec
    maxind = findmax(comparison)[2]
    minind = findmin(comparison)[2]
    lines!(ax_ecdf_examp, repeat([unif_vec[maxind] * 2*pi], 2), [unif_vec[maxind], ecdf[maxind]], 
        color=example_neuron_colors[examp_neuron], linewidth=1.5,
    )
    lines!(ax_ecdf_examp, repeat([unif_vec[minind] * 2*pi], 2), [unif_vec[minind], ecdf[minind]], 
        color=example_neuron_colors[examp_neuron], linewidth=1.5,
    )
    lines!(ax_ecdf_examp, xvec, ecdf, color=example_neuron_colors[examp_neuron], linewidth=2)
end
text!(ax_ecdf_examp, 5*pi/4, 0.4, text=L"D^-", color=example_neuron_colors["27"], fontsize=17)
text!(ax_ecdf_examp, 7*pi/4-pi/8, 0.9, text=L"D^+", color=example_neuron_colors["27"], fontsize=17)
textlabel!(ax_ecdf_examp, 1.6*pi, 0.1, text=L"V=D^+ + D^-", fontsize=17)

# Phase histograms ordered/sorted by circularity
ax_circ_sort = Axis(ge[2,1])
hidedecorations!(ax_circ_sort)
hidespines!(ax_circ_sort)
examp_circ = @subset(df_circ, :moth .== thismoth, string.(round.(Int, :neuron)) .∈ Ref(example_neurons))
examp_circ.neuron = string.(round.(Int, examp_circ.neuron))
sort!(examp_circ, [order(:kuiper_stat)])
circ_polar_axes = []
for (i, examp_neuron) in enumerate(examp_circ.neuron)
    this_inset = PolarAxis(ge[2,1], 
        width=Relative(0.4), height=Relative(0.4), halign=-0.5 + 0.4 * i, valign=0.5,
        clip=false, clip_r=false
    )
    hidedecorations!(this_inset)
    hist!(this_inset, phase_dict[examp_neuron] .* 2*pi, 
        normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[examp_neuron]))),
        color=example_neuron_colors[examp_neuron]
    )
    scatter!(this_inset, 0, 0, color=:black, markersize=5)
    push!(circ_polar_axes, this_inset)
end
# Put "axis" line, labels for Kuiper stat
xmin, xmax = 5, 40
ymin, ymax = -0.25, 0.8
arrows2d!(ax_circ_sort, Point2f(xmin, 0), [[xmax-xmin,0]])
ticks = Makie.get_tickvalues(LinearTicks(3), identity, xmin, xmax)
for lab_val in ticks
    lines!(ax_circ_sort, [lab_val, lab_val], [-0.04, 0.04], color=:black)
    text!(ax_circ_sort, lab_val, -0.06, text=string(lab_val), align=(:center, :top))
end
text!(ax_circ_sort, 0.5, 0.05, text="Kuiper statistic V", 
    space=:relative, align=(:center, :center))
limits!(ax_circ_sort, (xmin, xmax), (ymin, ymax))
let xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    on(events(f).window_open) do _  # fires once layout is resolved
        ax_p_bb = ax_circ_sort.layoutobservables.computedbbox[]
        ax_p_origin = ax_p_bb.origin
        ax_p_widths = ax_p_bb.widths
        for (pa, pval, neur) in zip(circ_polar_axes, examp_circ.kuiper_stat, examp_circ.neuron)
            pa_bb = pa.layoutobservables.computedbbox[]
            # Top of polar axis in scene pixels
            px_center = pa_bb.origin .+ [pa_bb.widths[1] / 2, pa_bb.widths[2]]
            # Convert scene pixels → ax_circ_sort data coords
            x_data = xmin + ((px_center[1] - ax_p_origin[1]) / ax_p_widths[1]) * (xmax - xmin)
            y_data = ymin + ((px_center[2] - pa_bb.widths[2]/2 - ax_p_origin[2]) / ax_p_widths[2]) * ymax  # will require fine tuning
            # Power position on number line
            color = example_neuron_colors[neur]
            lines!(ax_circ_sort, [x_data, pval], [y_data, 0], color=color, linewidth=0.8)
            sc = scatter!(ax_circ_sort, pval, 0, color=color, markersize=15)
            translate!(sc, 0, 0, 10)
            # Label position: centered on polar axis x, just above/below it
            x_label = xmin + ((px_center[1] - ax_p_origin[1]) / ax_p_widths[1]) * (xmax - xmin)
            # Label above neuron polar plot
            px_top = pa_bb.origin[2] + pa_bb.widths[2]
            y_label = ymin + ((px_top - ax_p_origin[2]) / ax_p_widths[2]) * ymax + 0.15
            text!(ax_circ_sort, x_label, y_label, text="N. " * neur,
                align=(:center, :bottom), fontsize=14, color=color, font=:bold)
        end
    end
end
lines!(ax_ecdf_examp, [0, 2*pi], [0, 1], color=:grey, linewidth=3.2) # Uniform distribution

for timing_info in ["Spike timing info", "No spike timing info"]
    gdf = @subset(df_circ, :has_timing_info .== timing_info)
    scatter!(ax_circ_mi, gdf.kuiper_stat, gdf.mi, 
        label=gdf.has_timing_info[1],
        color=timing_colors[gdf.has_timing_info[1]]
        # label=titlecase(direction), 
        # color=color_dict[direction]
    )
    if timing_info .== "Spike timing info"
        # mask = gdf.has_timing_info .== "Spike timing info"
        scatter!(ax_circ_prec, gdf.kuiper_stat, gdf.precision,
            color=timing_colors[gdf.has_timing_info[1]]
            # color=color_dict[direction]
        )
    end
end
axislegend(ax_circ_mi, position=:rb)
hidexdecorations!(ax_circ_mi, grid=false, minorgrid=false)

apply_letter_label(ge, "E")

# -------- Final panel showing how two methods disagree on neurons without timing information
dt = @pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all") |> 
leftjoin(_, dfc, on=[:moth, :neuron])

ax_stat_comp = Axis(gf[1,1], 
    xlabel="Kuiper statistic V", ylabel="Power at wingbeat frequency (dB/Hz)",
    xscale=log10,
    aspect=1
)
for gdf in groupby(dt, :has_timing_info)
    scatter!(ax_stat_comp, gdf.kuiper_stat, 10 .* log10.(gdf.peak_power), 
        label=gdf.has_timing_info[1], color=timing_colors[gdf.has_timing_info[1]])
end
axislegend(ax_stat_comp, position=:rb)
xlims!(ax_stat_comp, 0.86, 70)
apply_letter_label(gf, "F")

# Final adjustments 
colgap!(bottom_row, 20)
colsize!(bottom_row, 3, Relative(1/5))
rowsize!(f.layout, 2, Relative(0.5))
# return f
# end

# f = phasic_neuron_figure()
display(f)
save(joinpath(fig_dir, "fig_phasic.png"), f)

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