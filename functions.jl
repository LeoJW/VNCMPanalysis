# Function to load spike data from phy
# TODO: can potentially modify to include other details like cluster quality, doublet, etc
function read_phy_spikes(phydir)
    params = Dict(split(x, " = ")[1] => split(x, " = ")[2] for x in readlines(joinpath(phydir, "params.py")))
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
    return neurons
end

# Function to load spike data from AMPS
function get_amps_sort(path)
    if !("amps" in readdir(path))
        error("No amps directory found")
    end
    mat = readdlm(joinpath(path, "amps", "spikes.txt"), ',', Float32, '\n', skipstart=2)#, use_mmap=true)
    return mat
end

# Load digital barcodes from intan files


# Load digital barcodes from open-ephys

testpath = "/Volumes/PikesPeak/VNCMP/MP_data/good_data/2023-05-25_12-24-05/Record Node 101/experiment1/recording1"
# Function to read continuous stream(s?) from a given RecordNode/experiment/recording
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
    return time, data
end



# Utility functions 
function group_indices(vector)
    indices_dict = Dict{Int, Vector{Int}}()
    for (idx, value) in enumerate(vector)
        if haskey(indices_dict, value)
            push!(indices_dict[value], idx)
        else
            indices_dict[value] = [idx]
        end
    end
    return indices_dict
end