using DelimitedFiles
using NPZ
using JSON
using DataFrames
using DataFramesMeta
using DataStructures
using GLMakie

include("functions.jl")
include("functions_kinematics.jl")



moths = [
    "2025-02-25",
    "2025-02-25_1",
    "2025-03-11",
    "2025-03-12_1",
    "2025-03-20",
    "2025-03-21"
]
moths_kinematics = [
    "2025-02-25",
    "2025-02-25_1"
]
data_dir = "/Users/leo/Desktop/ResearchPhD/VNCMP/localdata"

moth = "2025-02-25_1"

fsamp = 30000
flap_amp_thresh = 0.4
flap_duration_thresh = 1000 # samples a bout must last

# Load open-ephys events, frametimes
# Special case for 2025-02-25_1 (has rep0 and rep1)
real_moth = length(split(moth, "_")) >= 3 ? moth[1:end-5] : moth
# Get open ephys dir as only dir without "sort" or "cam" in name
moth_dirs = first(walkdir(joinpath(data_dir, real_moth)))[2]
oedir = moth_dirs[findfirst((!).(occursin.("sort", moth_dirs)) .&& (!).(occursin.("cam", moth_dirs)))]

times, states, samples = read_events_open_ephys(joinpath(data_dir, real_moth, oedir))
oeps_times = times[states .== 3]
samples = samples[states .== 3]

# Find which frame event to start oeps events; usually the first one is a fluke, caused by arduino startup
# Special case for rep1 of moth 2025-02-25_1
if moth == "2025-02-25_1"
    second_start_ind = findlast(diff(oeps_times) .> 1)
    deleteat!(oeps_times, 1:second_start_ind)
    deleteat!(samples, 1:second_start_ind)
end
init_times_past_thresh = diff(oeps_times)[1:10] .> 0.01
start_ind = any(init_times_past_thresh) ? findfirst(init_times_past_thresh) + 1 : 1
oeps_times = oeps_times[start_ind:end]
samples = samples[start_ind:end]

# Read camera timestamps from frametimes.csv files
frametimes_dir = joinpath(data_dir, moth, "cameras")
df = [DataFrame() for i in 1:3]
for i in 0:2
    file = joinpath(frametimes_dir, "frametimes_" * string(i) * ".csv")
    df[i+1] = rename(CSV.read(file, DataFrame), ["frame", "time"])
end

# Find initial point where frametimes and open-ephys events agree within some tolerance
# Prior to this point, frame timestamps will be set from frametimes. 
# After this point, frame timestamps will be set from open ephys timestamps
oeps_ind = findfirst(diff(oeps_times[1:20000]) .> 1/350) # Drop to 300 Hz always happens in first 20 seconds
cam_ind = findfirst(diff(df[1].time[1:20000]) .> 1/350)
wi = 50
vec1 = df[1].time[cam_ind-wi:cam_ind+wi] .- df[1].time[1]
vec2 = oeps_times[oeps_ind-wi:oeps_ind+wi] .- oeps_times[1]
difference_mat = [x-y for x in vec1, y in vec2]
row_match = [argmin(abs.(row)) for row in eachrow(difference_mat)]
# Find where times start to match up 1:1 as longest run length encoding of diff 1
vals, lens = rle(diff(row_match))
initial_ind_cam = sum(lens[1:argmax(lens)-1]) + 1
initial_ind_oeps = row_match[initial_ind_cam]
# Adjust to indices of full vectors
initial_ind_cam += cam_ind - wi + 1
initial_ind_oeps += oeps_ind - wi + 1

# First part of frame indices from frametimes
offset_cam_to_oeps = oeps_times[initial_ind_oeps] - df[1].time[initial_ind_cam]
camera_frametimes = df[1].time[1:initial_ind_cam] .+ offset_cam_to_oeps 

frame_indices = round.(Int, camera_frametimes .* fsamp)

# Second part from arduino/open ephys timestamps
# Find point where desync becomes an issue, mark that point as cutoff
desync = [
    abs.(df[1].time .- df[2].time) .> 0.0125,
    abs.(df[2].time .- df[3].time) .> 0.0125,
    abs.(df[1].time .- df[3].time) .> 0.0125
]
desync_ind = any(any.(desync)) ? minimum(findfirst.(desync)) : nrow(df[1])
last_oeps_ind = searchsortedfirst(oeps_times, df[1].time[desync_ind] + offset_cam_to_oeps) - 1

frame_indices = vcat(frame_indices, samples[(initial_ind_oeps+1):last_oeps_ind])

# Read anipose/deeplabcut 3d kinematics up to file where desync happens, go no further
anipose_dir = joinpath(data_dir, "anipose_VNCMP", moth, "pose-3d")
kinematics_files = [f for f in readdir(anipose_dir) if occursin(".csv",f)]
file_inds = [parse(Int, split(f, "_")[1]) for f in kinematics_files]
sorti = sortperm(file_inds)
file_inds = file_inds[sorti]
kinematics_files = kinematics_files[sorti]
last_file_ind = findlast(file_inds .< desync_ind)
kdf = DataFrame()
for file in kinematics_files[1:last_file_ind]
    thisdf = CSV.read(joinpath(anipose_dir, file), DataFrame)
    for col in eachcol(thisdf)
        replace!(col, missing => NaN)
    end
    append!(kdf, thisdf)
end
# Keep only frames that were sync'd
kdf = kdf[1:length(frame_indices), :]
# Get point names Lhinge, Ltip, etc
point_cols = setdiff(
    names(kdf),
    ["M_00", "M_01", "M_02", "M_10", "M_11", "M_12", "M_20", "M_21", "M_22",
    "center_0", "center_1", "center_2", "fnum"]
)
point_names = [split(x,"_")[1] for x in point_cols[1:6:end]]


##


f = Figure()
ax = Axis(f[1,1])
ax2 = Axis(f[2,1])

# vlines!(ax, oeps_times .- (5399276 / 1000), ymin=0, ymax=0.5)
vlines!(ax, oeps_times, ymin=0, ymax=0.3)
vlines!(ax, df[1].time .+ offset_cam_to_oeps, ymin=0.3, ymax=0.6)
vlines!(ax, frame_indices ./ fsamp, ymin=0.6, ymax=1.0)
lines!(ax, df[1].time .+ offset_cam_to_oeps, vcat(0, diff(df[1].time)), color="black")


lines!(ax2, collect(1:nrow(kdf)) ./ fsamp, kdf.Ltip_x)

f

##

# times, states, samples = read_events_open_ephys(joinpath(data_dir, real_moth, oedir))
# oeps_times = times[states .== 3]

f = Figure()
ax = Axis(f[1,1])
vlines!(ax, times[states .== 3], ymin=0.5, ymax=1.0)
vlines!(ax, times[states .== -3], ymin=0.0, ymax=0.5)
f



##
using DSP

# Compute spectrogram
window_size = 1024
overlap = 64
spec = spectrogram(kdf.Ltip_x, window_size, overlap)

f, ax, hm = heatmap(spec.time, spec.freq, 10 .* log10.(transpose(spec.power)))


##


moth_dir = joinpath(data_dir, moth)

dir_contents = readdir(moth_dir)
phy_dir = joinpath(moth_dir, dir_contents[findfirst(occursin.("kilosort4", dir_contents))])
mp_dir = joinpath(moth_dir, dir_contents[findfirst(occursin.("_spikesort", dir_contents))])

neurons, unit_details, sort_params = read_phy_spikes(phy_dir)
muscles = read_mp_spikes(mp_dir)

fsamp = parse(Float64, sort_params["sample_rate"])
# Remove noise units
for (unit, quality) in unit_details["quality"]
    if quality == "noise"
        delete!(neurons, unit)
    end
end
# Remove spikes that are refractory period violations
for unit in keys(neurons)
    deleteat!(neurons[unit], findall(diff(neurons[unit]) .< (fsamp * refractory_thresh / 1000)) .+ 1)
end
# Write to npz file
# Remove data that isn't in a flapping region (by DLM timing)
# Get flapping periods
diff_vec = diff(muscles["ldlm"])
diff_over_thresh = findall(diff_vec .> duration_thresh .* fsamp)
start_inds = vcat(1, diff_over_thresh .+ 1)
end_inds = vcat(diff_over_thresh, length(muscles["ldlm"]))
# Get mean spike rate in each period, keep only those above a frequency and duration threshold
flap_duration = (muscles["ldlm"][end_inds] .- muscles["ldlm"][start_inds]) ./ fsamp
spike_rates = (end_inds - start_inds) ./ flap_duration
valid_periods = (flap_duration .> duration_thresh) .&& (spike_rates .> spike_rate_thresh)
# Grab valid periods, extend on either side by buffer seconds for neurons
bout_starts = max.(muscles["ldlm"][start_inds[valid_periods]] .- (buffer_in_sec * fsamp), 1)
bout_ends = muscles["ldlm"][end_inds[valid_periods]] .+ (buffer_in_sec * fsamp)
# Clear out spikes in neurons and muscles not valid by bouts
for unit in keys(neurons)
    valid = zeros(Bool, length(neurons[unit]))
    for i in eachindex(bout_starts)
        valid[(neurons[unit] .> bout_starts[i]) .&& (neurons[unit] .< bout_ends[i])] .= true
    end
    keepat!(neurons[unit], valid)
    # Remove anything empty (either doesn't overlap with muscles or slipped thru curation)
    if !any(valid)
        delete!(neurons, unit)
    end
end
for unit in keys(muscles)
    valid = zeros(Bool, length(muscles[unit]))
    for i in eachindex(bout_starts)
        valid[(muscles[unit] .> bout_starts[i]) .&& (muscles[unit] .< bout_ends[i])] .= true
    end
    keepat!(muscles[unit], valid)
    # Remove anything empty (either poor quality or slipped thru curation)
    if !any(valid)
        delete!(muscles, unit)
    end
end

data_dict = read_kinematics(moth; data_dir=data_dir)

##

f = Figure()
ax = Axis(f[1,1])
lines!(ax, data_dict["index"] ./ fsamp .- 180, data_dict["Rtheta"])
vlines!(ax, muscles["rdlm"] ./ fsamp, ymin=0, ymax=0.25)

f




## ---------- Look at kinematics against spike data


duration_thresh = 5 # Seconds long a flapping bout has to be
buffer_in_sec = 0.1 # Seconds on either side of a bout to keep. Must be less than duration_thresh/2 
spike_rate_thresh = 12 # Hz that a bout needs mean spike rate above
refractory_thresh = 1 # ms, remove spikes closer than this


moth = moths[2]
moth_dir = joinpath(data_dir, moth)
dir_contents = readdir(moth_dir)
phy_dir = joinpath(moth_dir, dir_contents[findfirst(occursin.("kilosort4", dir_contents))])
mp_dir = joinpath(moth_dir, dir_contents[findfirst(occursin.("_spikesort", dir_contents))])
neurons, unit_details, sort_params = read_phy_spikes(phy_dir)
muscles = read_mp_spikes(mp_dir)
fsamp = parse(Float64, sort_params["sample_rate"])

# Remove noise units
for (unit, quality) in unit_details["quality"]
    if quality == "noise"
        delete!(neurons, unit)
    end
end

data_dict = read_kinematics(moth; data_dir=data_dir)
# If this is 2025-02-25_1, also read and combine rep0
if moth == "2025-02-25_1"
    data_dict_rep0 = read_kinematics(moth * "_rep0"; data_dir=data_dir)
    data_dict = Dict(key => vcat(data_dict_rep0[key], data_dict[key]) for key in keys(data_dict))
end

f, ax, ln = lines(data_dict["index"] ./ fsamp, data_dict["Rtheta"])
vlines!(ax, muscles["rdlm"] ./ fsamp, ymin=0, ymax=0.3)
f