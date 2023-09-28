using NPZ   # Read .npy files
using JSON  # Read json formatted files, duh
using DelimitedFiles
using Mmap
include("IntanReader.jl")

moths = ["2023-05-20", "2023-05-25"]


vnc_dir = "/Volumes/PikesPeak/VNCMP"
motor_program_dir = "/Volumes/PikesPeak/VNCMP/MP_data/good_data"


# Time synchronize and match data from intan, open-ephys, and spike sorted sources


""" 
Function to match data for one individual moth
Takes two directory locations, parent dir of intan neural recordings and 
parent dir of motor program open-ephys recordings
"""
function read_and_match_moth_data(vnc_dir, mp_dir)
    # Get how many "pokes" or intan folder runs there are
    poke_dirs = [str for str in readdir(vnc_dir) if !any(x -> contains(str, x), [".", "noise", "test", "phy"])]
    # Get matching Phy dirs for each poke
    if !isdir(joinpath(vnc_dir, "phy_folder"))
        println("No folder named 'Phy Folder' found")
        return
    end
    phy_dirs = [str for str in readdir(joinpath(vnc_dir, "phy_folder")) if !any(contains(str, "."))]
    if isempty(phy_dirs)
        println("No Phy folders detected for moth " * splitpath(vnc_dir)[end])
        return
    end
    # Keep only pokes with phy dirs
    poke_dirs = [poke for poke in poke_dirs if any(contains.(phy_dirs, poke))]
    # Get creation time of all files in each poke directory, determine time order of "pokes"
    rhd_files = Dict{String, Vector{String}}()
    rhd_times = Dict{String, Vector{Float64}}()
    for poke in poke_dirs
        rhd_files[poke] = [x for x in readdir(joinpath(vnc_dir, poke)) if contains(x, ".rhd")]
        rhd_times[poke] = [ctime(joinpath(vnc_dir, poke, x)) for x in rhd_files[poke]]
    end
    poke_dirs = poke_dirs[sortperm([minimum(rhd_times[x]) for x in poke_dirs])]

    # Split out motor program folders
    mp_contents = readdir(mp_dir)
    amps_dir = joinpath(mp_dir, mp_contents[findfirst(occursin.("_spikesort", mp_contents))])
    openephys_dir = joinpath(mp_dir, mp_contents[findfirst(occursin.("Record Node", mp_contents))])
    # Match open-ephys experiments to each poke
    # Get open-ephys experiments, get start time in UTF
    experiments = [s for s in readdir(openephys_dir) if occursin("experiment", s)]
    oep_times = [readdlm(joinpath(openephys_dir, s, "recording1", "sync_messages.txt"))[1,10] for s in experiments]
    oep_times = trunc.(Int, oep_times / 1000) # ms to s

    # Load AMPS data
    amps_mat = get_amps_sort(amps_dir)

    # For each experiment, use universal timestamps to guess which .rhd file in associated poke matches start time of experiment
    for (i, exp) in enumerate(experiments)
        # Find which rhd files happened after this experiment started
        time_compare = [findfirst(rhd_times[p] .> oep_times[i]) for p in poke_dirs]
        poke_match_ind = findfirst((!).(isnothing.(time_compare)))
        rhd_match_ind = time_compare[poke_match_ind] - 1
        # If open-ephys was started before intan index will be zero
        # Conditional here so I can note those times
        if rhd_match_ind == 0
            rhd_match_ind += 1
        end
        # Get all AMPS data for this experiment
        trial_start_times = readdlm(joinpath(amps_dir, "trial_start_times.txt"), ',', Any, '\n', header=true)[1]
        exp_num = [parse(Int, split(x, "_")[2][end]) for x in trial_start_times[:,1]] .+ 1
        trials_in_this_experiment = findall(exp_num .== parse(Int, exp[end]))
        amps_trial_inds = vec(any(amps_mat[:,1] .== trials_in_this_experiment', dims=2))
        mp_spike_inds = Int.(amps_mat[amps_trial_inds,4])
        # Load run of .rhd files and open-ephys data where we have AMPS spikes

        # Come up with sample mapping based on barcodes


        # exp_to_poke[exp] = poke_dirs[findfirst(time_match)]
        return trial_start_times, mp_spike_inds
    end
end

trial_start_times, mp_spike_inds = read_and_match_moth_data(
    "/Volumes/PikesPeak/VNCMP/2023-05-25", 
    "/Volumes/PikesPeak/VNCMP/MP_data/good_data/2023-05-25_12-24-05")


# Function to load barcodes from intan files

# Function to load barcodes from open-ephys 

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
    mat = readdlm(joinpath(path, "amps", "spikes.txt"), ',', Float32, '\n', skipstart=2, use_mmap=true)
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