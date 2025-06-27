# Function to load spike data from phy
function read_phy_spikes(phydir)
    # Sorting parameters
    sort_params = Dict(split(x, " = ")[1] => split(x, " = ")[2] for x in readlines(joinpath(phydir, "params.py")))
    # Read actual unit data
    spike_inds = npzread(joinpath(phydir, "spike_times.npy"))
    units = npzread(joinpath(phydir, "spike_clusters.npy"))
    neurons = Dict{Int, Vector{Int64}}()
    for (idx, value) in enumerate(units)
        if haskey(neurons, value)
            push!(neurons[value], spike_inds[idx])
        else
            neurons[value] = [spike_inds[idx]]
        end
    end
    # Information on unit quality
    unit_details = Dict{String, Dict{Int, Any}}(
        "quality" => Dict(key => "unk" for key in keys(neurons)),
        "doublet" => Dict(key => false for key in keys(neurons)))
    unit_quality = readdlm(joinpath(phydir, "cluster_group.tsv"), skipstart=1)
    for row in eachrow(unit_quality)
        unit_details["quality"][row[1]] = row[2]
    end
    # Add information on doublets if provided
    if isfile(joinpath(phydir, "cluster_doublet.tsv"))
        unit_doublets = readdlm(joinpath(phydir, "cluster_doublet.tsv"), skipstart=1)
        for unit in keys(neurons)
            if unit in unit_doublets[:,1]
                unit_details["doublet"][unit] = true
            end
        end
    end
    return neurons, unit_details, sort_params
end

# Function to load spike data from AMPS
function get_amps_sort(path)
    if !("amps" in readdir(path))
        error("No amps directory found")
    end
    mat = readdlm(joinpath(path, "amps", "spikes.txt"), ',', Float32, '\n', skipstart=2)#, use_mmap=true)
    return mat
end

# Function like read_phy_spikes, to read in motor program spikes and shift back to real times based on trial_start_times
# Picks max experiment number if use_exp is negative (default)
function read_mp_spikes(mp_dir::String; use_exp::Int = -1)
    muscle_code = Dict(
        0 => "lax", 1 => "lba", 2 => "lsa", 3 => "ldvm", 4 => "ldlm",
        5 => "rdlm", 6 => "rdvm", 7 => "rsa", 8 => "rba", 9 => "rax")
    amps_mat = get_amps_sort(mp_dir)
    trial_start_times = readdlm(joinpath(mp_dir, "trial_start_times.txt"), ',', Any, '\n', header=true)[1]
    exp_num = [parse(Int, split(x, "_")[2][end]) for x in trial_start_times[:,1]] .+ 1
    if use_exp < 0
        use_exp = maximum(exp_num)
    end
    trial_start_ind = Dict{Int, Int}(parse(Int, split(x[1], "_")[3][1:3]) => x[2] for x in eachrow(trial_start_times))
    trials_in_this_experiment = findall(exp_num .== use_exp)
    amps_trial_inds = vec(any(amps_mat[:,1] .== trials_in_this_experiment', dims=2))
    mask = amps_trial_inds .&& (amps_mat[:,6] .== 1) # get spikes in this experiment also marked as valid
    mp_spike_inds = Int.(amps_mat[mask,4])
    # Shift spike indices by starting index of each trial, +1 because moving from 0-index python to 1-index julia
    mp_spike_inds = mp_spike_inds .+ [trial_start_ind[x] for x in amps_mat[mask,1]] .+ 1
    # Populate output dict
    muscles = Dict{String, Vector{Int64}}()
    for (muscle_int, muscle_name) in muscle_code
        muscles[muscle_name] = mp_spike_inds[amps_mat[mask,2] .== muscle_int]
    end
    return muscles
end

# Function to read continuous stream(s?) from a given RecordNode/experiment/recording. Must pass in experiment folder!
# TODO: Make more flexible (or make wrapper function) to catch being passed recording node, experiment, or top-level dir
function read_binary_open_ephys(recording_folder)
    dir_contents = readdir(recording_folder)
    # Find and read structure.oebin
    structure_file = dir_contents .== "structure.oebin"
    if !any(structure_file)
        println("WARNING: No structure.oebin file found, exiting without read")
        return
    end
    oebin = JSON.parsefile(joinpath(recording_folder, dir_contents[findfirst(structure_file)]))
    # NOTE: The [1] just grabs whichever continuous stream is first. Won't work with more than 1 stream
    continuous_dir = joinpath(recording_folder, "continuous", oebin["continuous"][1]["folder_name"])
    # Read time first
    time = npzread(joinpath(continuous_dir, "timestamps.npy"))
    # Read all channels into matrix
    N = length(time)
    num_channels = oebin["continuous"][1]["num_channels"]
    data = zeros(Float64, N, num_channels)
    # NOTE: open-ephys binary format uses Int16 in little endian. Not sure this will work on any system
    fid = open(joinpath(continuous_dir, "continuous.dat"))
    datamap = mmap(fid, Matrix{Int16}, (num_channels, N))
    for (i,ch) in enumerate(oebin["continuous"][1]["channels"])
        data[:,i] .= datamap[i,:] .* ch["bit_volts"]
    end
    close(fid)
    # First entry of channels is somehow always zero. Set to identical to 2nd value to make smoother
    data[1,:] = data[2,:]
    return time, data
end
function read_binary_open_ephys(recording_folder, stream_indices)
    dir_contents = readdir(recording_folder)
    # Find and read structure.oebin
    structure_file = dir_contents .== "structure.oebin"
    if !any(structure_file)
        println("WARNING: No structure.oebin file found, exiting without read")
        return
    end
    oebin = JSON.parsefile(joinpath(recording_folder, dir_contents[findfirst(structure_file)]))
    # NOTE: The [1] just grabs whichever continuous stream is first. Won't work with more than 1 stream
    continuous_dir = joinpath(recording_folder, "continuous", oebin["continuous"][1]["folder_name"])
    # Read time first
    time = npzread(joinpath(continuous_dir, "timestamps.npy"))
    # Read all channels into matrix
    N = length(time)
    num_channels = oebin["continuous"][1]["num_channels"]
    data = zeros(Float64, N, length(stream_indices))
    # NOTE: open-ephys binary format uses Int16 in little endian. Not sure this will work on any system
    fid = open(joinpath(continuous_dir, "continuous.dat"))
    datamap = mmap(fid, Matrix{Int16}, (num_channels, N))
    for (i,chi) in enumerate(stream_indices)
        data[:,i] .= datamap[chi,:] .* oebin["continuous"][1]["channels"][chi]["bit_volts"]
    end
    close(fid)
    # First entry of channels is somehow always zero. Set to identical to 2nd value to make smoother
    data[1,:] = data[2,:]
    return time, data
end




# Assumes oep_events and rhd_events already store the first known event
function match_events!(
    oep_events, rhd_events, # storage arrays for matching events
    oep_indices, rhd_indices, # indices of all events to find matches in
    oep_first_match, rhd_first_match; # initial first known match
    frame_dif_tol=10,
    check_n_events=15)
    # Kinda sloppy to work backwards then forwards. But I have to write this quickly! Easiest thing that sprang to mind
    # Work backwards from first match
    oep_index, rhd_index = oep_first_match, rhd_first_match
    while true
        if oep_index <= 2 || rhd_index <= 2
            break
        end
        oep_dif = oep_indices[oep_index] - oep_indices[oep_index-1]
        rhd_dif = rhd_indices[rhd_index] - rhd_indices[rhd_index-1]
        # Happy path where previous frames match up close enough in time
        if abs(oep_dif - rhd_dif) < frame_dif_tol
            append!(oep_events, oep_indices[oep_index-1])
            append!(rhd_events, rhd_indices[rhd_index-1])
            oep_index -= 1
            rhd_index -= 1
        # If next frame down doesn't match, find wherever next match is
        else
            # Check last X events
            back_ind = min(check_n_events, min(oep_index - 1, rhd_index - 1))
            oep_frames = oep_indices[oep_index] .- oep_indices[oep_index-back_ind:oep_index-1]
            rhd_frames = rhd_indices[rhd_index] .- rhd_indices[rhd_index-back_ind:rhd_index-1]
            has_matches = [any(abs.(x .- rhd_frames) .< frame_dif_tol) for x in oep_frames]
            # Kill loop if no matches, otherwise jump to last one with match
            if !any(has_matches)
                break
            end
            oep_index = oep_index - back_ind + findlast(has_matches) - 1
            rhd_index = rhd_index - back_ind + findfirst(abs.(oep_frames[findlast(has_matches)] .- rhd_frames) .< frame_dif_tol) - 1
            append!(oep_events, oep_indices[oep_index])
            append!(rhd_events, rhd_indices[rhd_index])
        end
    end
    # Work forwards from first match
    oep_index, rhd_index = oep_first_match, rhd_first_match
    while true
        if oep_index >= length(oep_indices) - 1 || rhd_index >= length(rhd_indices) - 1
            break
        end
        oep_dif = oep_indices[oep_index+1] - oep_indices[oep_index]
        rhd_dif = rhd_indices[rhd_index+1] - rhd_indices[rhd_index]
        # Happy path where previous frames match up close enough in time
        if abs(oep_dif - rhd_dif) < frame_dif_tol
            append!(oep_events, oep_indices[oep_index+1])
            append!(rhd_events, rhd_indices[rhd_index+1])
            oep_index += 1
            rhd_index += 1
        # If next frame up doesn't match, find wherever next match is
        else
            # Check next X events
            next_ind = min(check_n_events, min(length(oep_indices) - oep_index, length(rhd_indices) - rhd_index))
            oep_frames = oep_indices[oep_index+1:oep_index+next_ind] .- oep_indices[oep_index]
            rhd_frames = rhd_indices[rhd_index+1:rhd_index+next_ind] .- rhd_indices[rhd_index]
            has_matches = [any(abs.(x .- rhd_frames) .< frame_dif_tol) for x in oep_frames]
            # Kill loop if no matches, otherwise jump to last one with match
            if !any(has_matches)
                break
            end
            oep_index = oep_index + findfirst(has_matches)
            rhd_index = rhd_index + findfirst(abs.(oep_frames[findfirst(has_matches)] .- rhd_frames) .< frame_dif_tol)
            append!(oep_events, oep_indices[oep_index])
            append!(rhd_events, rhd_indices[rhd_index])
        end
    end
end


"""
Remove duplicate spikes. 
Spikes are considered duplicated if they are less than x ms apart where x is the exclusion period
Expects dictionary of units, where each entry is an array of ints of spike event indices
Keeps first spike. 
If multiple adjacent spikes are too close, will remove all but first, even if removing a middle one would resolve
"""
function remove_duplicate_spikes!(neurons; exclusion_period_ms=1.0, fsamp=30000)
    exclusion_period_samples = round(Int, exclusion_period_ms / 1000 * fsamp)
    for key in keys(neurons)
        too_close_inds = findall(diff(neurons[key]) .<= exclusion_period_samples) .+ 1
        deleteat!(neurons[key], too_close_inds)
    end
end

"""
unwrap_spikes
Takes a set of columns, and "unwraps" spikes by assigning ones before or after a phase threshold
to adjacent wingstrokes. This allows us to group muscle firing as bursts without discontinuities, 
as bursts may overlap the arbitrary point at which we cut up wingstrokes and get separated

Different from comparative version. 

Moves everything to next wingbeat, so always generates negative phases instead of phase > 1.0
NOTE: phase_wrap_thresholds currently used as a global variable
"""
function unwrap_spikes_to_next(time, phase, wb, wblen, muscle, ismuscle)
    wbi = group_indices(wb)
    # For each muscle
    for m in unique(muscle[ismuscle])
        threshold = phase_wrap_thresholds[m[2:end]]
        # Wrap spikes past threshold to next wingbeat
        # Loop over each wingbeat
        for i in eachindex(wbi)
            # Get indices for this muscle in this wingbeat, move on if nothing
            mi = findall(muscle[wbi[i]] .== m)
            if length(mi) == 0
                continue
            end
            # Get which spikes in this wb are past threshold
            inds = findall(phase[wbi[i][mi]] .>= threshold)
            # Jump to next wingbeat if no spikes need to move
            if length(inds) == 0
                continue
            end
            inds = wbi[i][mi][inds]
            if haskey(wbi, i+1)
                # Order of these lines matters a lot! Time shift uses current wblen, but phase shift uses next
                time[inds] .-= wblen[inds] # time = -(wblen - time)
                wblen[inds] .= first(wblen[wbi[i+1]])
                phase[inds] = time[inds] ./ wblen[inds]
                wb[inds] .+= 1
            # If next wingbeat doesn't exist, mark to remove these spikes
            # (Spike count and info theo analyses require complete wingbeats)
            else
                time[inds] .= NaN
            end
        end
    end
    return DataFrame(time=time, phase=phase, wb=wb, wblen=wblen)
end
function unwrap_spikes_to_prev(time, phase, wb, wblen, muscle, ismuscle)
    wbi = group_indices(wb)
    # For each muscle
    for m in unique(muscle[ismuscle])
        threshold = phase_wrap_thresholds[m[2:end]]
        # Wrap spikes past threshold to next wingbeat
        # Loop over each wingbeat
        for i in eachindex(wbi)
            # Get indices for this muscle in this wingbeat, move on if nothing
            mi = findall(muscle[wbi[i]] .== m)
            if length(mi) == 0
                continue
            end
            # Get which spikes in this wb are past threshold
            inds = findall(phase[wbi[i][mi]] .< threshold)
            # Jump to next wingbeat if no spikes need to move
            if length(inds) == 0
                continue
            end
            inds = wbi[i][mi][inds]
            # Get wblen to use for shifted spikes
            if haskey(wbi, i-1)
                wblen[inds] .= first(wblen[wbi[i-1]])
                time[inds] .+= wblen[inds]
                phase[inds] = time[inds] ./ wblen[inds]
                wb[inds] .-= 1
            # If prev wingbeat doesn't exist, mark to remove these spikes
            # (Spike count and info theo analyses require complete wingbeats)
            else
                time[inds] .= NaN
            end
        end
    end
    return DataFrame(time=time, phase=phase, wb=wb, wblen=wblen)
end


#---- Utility functions 
"""
Utility function for quick finding of indices for all unique values
Works exactly the same as StatsBase.countmap, probably best to just use that
"""
function group_indices(input)
    indices_dict = Dict{eltype(input), Vector{eltype(input)}}()
    for (idx, value) in enumerate(input)
        if haskey(indices_dict, value)
            push!(indices_dict[value], idx)
        else
            indices_dict[value] = [idx]
        end
    end
    return indices_dict
end

function find_threshold_crossings(signal, threshold)
    crossings = Int[]
    above_threshold = signal[1] > threshold ? true : false
    for (index, value) in enumerate(signal)
        if !above_threshold && value > threshold
            push!(crossings, index)
            above_threshold = true
        elseif above_threshold && value <= threshold
            above_threshold = false
        end
    end
    return crossings
end
function find_threshold_crossings(signal, threshold, debounce_window::Int)
    crossings = Int[]
    above_threshold = signal[1] > threshold ? true : false
    ind = -100
    for (index, value) in enumerate(signal)
        if !above_threshold && value > threshold && (index - ind) > debounce_window
            push!(crossings, index)
            above_threshold = true
            ind = index  # Update the debounce tracking index for both directions
        elseif above_threshold && value <= threshold && (index - ind) > debounce_window
            above_threshold = false
            ind = index  # Update the debounce tracking index for both directions
        end
    end
    return crossings
end

function find_common_elements(vec1, vec2)
    common = Int[]  # Initialize an empty array to store common elements
    i, j = 1, 1  # Initialize pointers for both vectors
    while i <= length(vec1) && j <= length(vec2)
        if vec1[i] == vec2[j]
            push!(common, vec1[i])  # Both vectors have the same element, so add it to the common array
            i += 1
            j += 1
        elseif vec1[i] < vec2[j]
            i += 1  # Move the pointer in the first vector
        else
            j += 1  # Move the pointer in the second vector
        end
    end
    return common
end

"""
Find maximum length run of repeating elements in a vector
"""
function max_count(vec::Vector)
    max_count = 0
    current_count = 1
    for i in 2:length(vec)
        if vec[i] == vec[i-1]
            current_count += 1
        else
            current_count = 1
        end
        max_count = max(max_count, current_count)
    end
    return max_count
end