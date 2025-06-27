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

duration_thresh = 5 # Seconds long a flapping bout has to be
buffer_in_sec = 0.1 # Seconds on either side of a bout to keep. Must be less than duration_thresh/2 
spike_rate_thresh = 12 # Hz that a bout needs mean spike rate above
refractory_thresh = 1 # ms, remove spikes closer than this

##
for moth in moths
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
    
    # Assemble export dict
    output_dict = Dict{String, Vector{Int64}}()
    label_dict = Dict{String, Int}()
    units_vec = vcat(string.(collect(keys(neurons))), collect(keys(muscles)))
    for unit in units_vec
        if occursin(r"[0-9]", unit)
            output_dict[unit] = neurons[parse(Int,unit)]
            # label_dict[unit] = unit_details["quality"][parse(Int,unit)]
            label_dict[unit] = unit_details["quality"][parse(Int,unit)] == "good" ? 1 : 0
        else
            output_dict[unit] = muscles[unit]
            label_dict[unit] = 2
        end
    end
    
    npzwrite(joinpath(data_dir, moth * "_data.npz"), output_dict)
    npzwrite(joinpath(data_dir, moth * "_labels.npz"), label_dict)
    npzwrite(joinpath(data_dir, moth * "_bouts.npz"), Dict("starts" => bout_starts, "ends" => bout_ends))
end

## Read and save kinematics 
for moth in moths_kinematics
    data_dict = read_kinematics(moth; data_dir=data_dir)
    # If this is 2025-02-25_1, also read and combine rep0
    if moth == "2025-02-25_1"
        data_dict_rep0 = read_kinematics(moth * "_rep0"; data_dir=data_dir)
        data_dict = Dict(key => vcat(data_dict_rep0[key], data_dict[key]) for key in keys(data_dict))
    end
    npzwrite(joinpath(data_dir, moth * "_kinematics.npz"), data_dict)
end


##

moth = moths[3]
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

## Plot for fun

good_units = [unit for (unit, qual) in unit_details["quality"] if qual == "good"]
mua_units = [unit for (unit, qual) in unit_details["quality"] if qual == "mua"]

##
f = Figure()
ax = Axis(f[1,1])
seg = 1 / (length(good_units)+1)
for (i,unit) in enumerate(good_units)
    vlines!(ax, neurons[unit] ./ fsamp, ymin=i*seg, ymax=(i+1)*seg)
end
vlines!(ax, muscles["ldlm"] ./ fsamp, ymin=0.0, ymax=seg)
f
##
f = Figure()
ax = Axis(f[1,1])
vlines!(ax, neurons[57] ./ fsamp, ymin=0.5, ymax=1.0)
vlines!(ax, muscles["ldlm"] ./ fsamp, ymin=0.0, ymax=0.5)
f
##
for unit in keys(neurons)
    deleteat!(neurons[unit], findall(diff(neurons[unit]) .< (fsamp * refractory_thresh / 1000)) .+ 1)
end
##
unit = rand(good_units)
dif = diff(neurons[unit]) ./ fsamp .* 1000
f, ax, hs = hist(dif[dif .< 10], bins=100)
dif = dif[dif .> 1]
hist!(ax, dif[dif .< 10], bins=100)
f
