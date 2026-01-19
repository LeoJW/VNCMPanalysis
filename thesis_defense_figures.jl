using HDF5
using CSV
using NPZ
using DelimitedFiles
using Statistics
using StatsBase # Mainly for rle
using GLM
using GLMakie
using CairoMakie
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
##

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

##

neurons, muscles, unit_details = get_spikes("2025-03-12_1")

## -------------------------------- Zoom in on data to get to figure 1

embedding = h5read(joinpath(data_dir, "..", "fig_1_embedding_data.h5"), "dict_0")

# Define target zoom range
target_range = [1101, 1101.5]

# Calculate full data range for starting zoom
full_time_range = [minimum(embedding["time"]), maximum(embedding["time"])]

fsamp = 30000
good_units = [unit for (unit, qual) in unit_details["quality"] if qual == "good"]
# good_units = [unit for (unit, qual) in unit_details["quality"] if qual != "noise"]
good_muscles = [m for m in single_muscles if length(muscles[m]) > 0]

# Pre-calculate unit firing rates and sorting (using target range for consistency)
unit_fr = [mean(diff(neurons[unit][(neurons[unit] .> target_range[1] * fsamp) .&& (neurons[unit] .< target_range[2] * fsamp)])) for unit in good_units]
# unit_fr = [mean(diff(neurons[unit])) for unit in good_units]
nspikes = [length(neurons[unit]) for unit in good_units]
mask = (!).(isnan.(unit_fr))
sorti = sortperm(DataFrame(a=unit_fr[mask], b=nspikes[mask], copycols=false))
# sorti = sortperm(DataFrame(b=nspikes[mask], copycols=false))

n_frames = 200
frames = Vector{Matrix{RGB{Float64}}}()

# Smooth easing function (ease-out cubic)
function ease_out_cubic(t)
    return 1 - (1 - t)^3
end

# Alternative easing functions you can try:
# Ease-out quadratic (gentler): t -> 1 - (1 - t)^2
# Ease-out exponential (stronger): t -> t == 1 ? 1 : 1 - 2^(-10 * t)
# Ease-out sine (smooth): t -> sin(t * π / 2)

zoom_scale = [logrange(full_time_range[1], target_range[1], n_frames), logrange(full_time_range[2], target_range[2], n_frames)]
# Create smooth zoom transition
t_values = range(0, 1, length=n_frames)
eased_t = ease_out_cubic.(t_values)
# Interpolate between full range and target range using easing
zoom_scale = [
    full_time_range[1] .+ eased_t .* (target_range[1] - full_time_range[1]),
    full_time_range[2] .+ eased_t .* (target_range[2] - full_time_range[2])
]

# Animation loop
for frame_i in 1:n_frames
    plot_range = [zoom_scale[1][frame_i], zoom_scale[2][frame_i]]
    use_xticks = [plot_range[1], plot_range[1] + (plot_range[2] - plot_range[1]) / 2, plot_range[2]]

    f = Figure(size=(900,980))
    ax = Axis(f[1,1], xticks=use_xticks, yticks=([],[]))
    axe = Axis(f[2,1], xticks=use_xticks, yticks=([],[]))
    axm = Axis(f[3,1], xticks=use_xticks, yticks=([],[]), xlabel="Time (s)")

    # Plot neural units
    seg = 1 / length(good_units[mask][sorti])
    for (i, unit) in enumerate(good_units[mask][sorti])
        # # Skip some upper rows for empty space
        # if (i >= 13) && (i <= 13 + 3)
        #     continue
        # end
        # col = unit == 97 ? Makie.wong_colors()[2] : :black # 33 also good
        col = :black
        # Filter spikes to current plot range
        unit_spikes = neurons[unit] ./ fsamp
        visible_spikes = unit_spikes[(unit_spikes .>= plot_range[1]) .&& (unit_spikes .<= plot_range[2])]
        if !isempty(visible_spikes)
            vlines!(ax, visible_spikes, ymin=(i-1)*seg, ymax=i*seg, color=col)
        end
    end
    
    # Plot muscle activity
    seg = 1 / length(good_muscles)
    for (i, unit) in enumerate(good_muscles)
        muscle_spikes = muscles[unit] ./ fsamp
        visible_spikes = muscle_spikes[(muscle_spikes .>= plot_range[1]) .&& (muscle_spikes .<= plot_range[2])]
        if !isempty(visible_spikes)
            vlines!(axm, visible_spikes, ymin=(i-1)*seg, ymax=i*seg, color=muscle_colors_dict[unit])
        end
    end

    # # Plot embedding data
    # time_mask = (embedding["time"] .>= plot_range[1]) .&& (embedding["time"] .<= plot_range[2])
    # if any(time_mask)
    #     for i in 1:3
    #         xvals = embedding["X"][i, time_mask]
    #         yvals = embedding["Y"][i, time_mask]
    #         if !isempty(xvals)
    #             xvals = (xvals .- mean(xvals)) ./ std(xvals)
    #             yvals = (yvals .- mean(yvals)) ./ std(yvals)
    #             lines!(axe, embedding["time"][time_mask], xvals .+ 6*i, color=Makie.wong_colors()[2])
    #             lines!(axe, embedding["time"][time_mask], yvals .+ 6*i, color=Makie.wong_colors()[1])
    #         end
    #     end
    # end

    # Link axes and set properties
    linkxaxes!(ax, axe, axm)
    hidedecorations!(ax)
    hidedecorations!(axe)
    hideydecorations!(axm, ticks=false)
    hidespines!(axe)
    hidespines!(ax)
    hidespines!(axm)
    xlims!(ax, plot_range)
    xlims!(axe, plot_range)
    xlims!(axm, plot_range)
    rowsize!(f.layout, 1, Auto(1.5))
    rowsize!(f.layout, 3, Auto(1.2))
    # Convert figure to image and add to frames
    img = Makie.colorbuffer(f)
    push!(frames, img)
    # Optional: print progress
    if frame_i % 10 == 0
        println("Frame $frame_i/$n_frames completed")
    end
end

# Save as GIF
using FileIO
save(joinpath(fig_dir, "zoom_animation.gif"), cat(frames..., dims=3); fps=20)


##------------------------ Highlight neuron on last frame


embedding = h5read(joinpath(data_dir, "..", "fig_1_embedding_data.h5"), "dict_0")

# Define target zoom range
target_range = [1101, 1101.5]

# Calculate full data range for starting zoom
full_time_range = [minimum(embedding["time"]), maximum(embedding["time"])]

fsamp = 30000
good_units = [unit for (unit, qual) in unit_details["quality"] if qual == "good"]
# good_units = [unit for (unit, qual) in unit_details["quality"] if qual != "noise"]
good_muscles = [m for m in single_muscles if length(muscles[m]) > 0]

# Pre-calculate unit firing rates and sorting (using target range for consistency)
unit_fr = [mean(diff(neurons[unit][(neurons[unit] .> target_range[1] * fsamp) .&& (neurons[unit] .< target_range[2] * fsamp)])) for unit in good_units]
# unit_fr = [mean(diff(neurons[unit])) for unit in good_units]
nspikes = [length(neurons[unit]) for unit in good_units]
mask = (!).(isnan.(unit_fr))
sorti = sortperm(DataFrame(a=unit_fr[mask], b=nspikes[mask], copycols=false))
# sorti = sortperm(DataFrame(b=nspikes[mask], copycols=false))

n_frames = 200
frames = Vector{Matrix{RGB{Float64}}}()

# Animation loop

plot_range = [1101, 1101.5]
use_xticks = [plot_range[1], plot_range[1] + (plot_range[2] - plot_range[1]) / 2, plot_range[2]]

f = Figure(size=(900,980))
ax = Axis(f[1,1], xticks=(use_xticks, ["0", "0.25", "0.5"]), yticks=([],[]))
axe = Axis(f[2,1], xticks=(use_xticks, ["0", "0.25", "0.5"]), yticks=([],[]))
axm = Axis(f[3,1], xticks=(use_xticks, ["0", "0.25", "0.5"]), yticks=([],[]), xlabel="Time (s)")

# Plot neural units
seg = 1 / length(good_units[mask][sorti])
for (i, unit) in enumerate(good_units[mask][sorti])
    # # Skip some upper rows for empty space
    # if (i >= 13) && (i <= 13 + 3)
    #     continue
    # end
    col = unit == 33 ? Makie.wong_colors()[2] : RGBf(0.8, 0.8, 0.8) # 33 also good
    # Filter spikes to current plot range
    unit_spikes = neurons[unit] ./ fsamp
    visible_spikes = unit_spikes[(unit_spikes .>= plot_range[1]) .&& (unit_spikes .<= plot_range[2])]
    if !isempty(visible_spikes)
        vlines!(ax, visible_spikes, ymin=(i-1)*seg, ymax=i*seg, color=col, linewidth=3)
    end
end
# Plot muscle activity
seg = 1 / length(good_muscles)
for (i, unit) in enumerate(good_muscles)
    muscle_spikes = muscles[unit] ./ fsamp
    visible_spikes = muscle_spikes[(muscle_spikes .>= plot_range[1]) .&& (muscle_spikes .<= plot_range[2])]
    if !isempty(visible_spikes)
        vlines!(axm, visible_spikes, ymin=(i-1)*seg, ymax=i*seg, color=muscle_colors_dict[unit], linewidth=3)
    end
end

# # Plot embedding data
# time_mask = (embedding["time"] .>= plot_range[1]) .&& (embedding["time"] .<= plot_range[2])
# if any(time_mask)
#     for i in 1:3
#         xvals = embedding["X"][i, time_mask]
#         yvals = embedding["Y"][i, time_mask]
#         if !isempty(xvals)
#             xvals = (xvals .- mean(xvals)) ./ std(xvals)
#             yvals = (yvals .- mean(yvals)) ./ std(yvals)
#             lines!(axe, embedding["time"][time_mask], xvals .+ 6*i, color=Makie.wong_colors()[2])
#             lines!(axe, embedding["time"][time_mask], yvals .+ 6*i, color=Makie.wong_colors()[1])
#         end
#     end
# end

# Link axes and set properties
linkxaxes!(ax, axe, axm)
hidedecorations!(ax)
hidedecorations!(axe)
hideydecorations!(axm, ticks=false)
hidespines!(axe)
hidespines!(ax)
hidespines!(axm)
xlims!(ax, plot_range)
xlims!(axe, plot_range)
xlims!(axm, plot_range)
rowsize!(f.layout, 1, Auto(1.5))
rowsize!(f.layout, 3, Auto(1.2))

save(joinpath(fig_dir, "fig1_last_panel_neuron_highlighted.png"), f)