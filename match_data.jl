using NPZ   # Read .npy files
using JSON  # Read json formatted files, duh
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
    rhd_times = Dict{String, Vector{Float64}}()
    for poke in poke_dirs
        rhd_times[poke] = [ctime(joinpath(vnc_dir, poke, x)) for x in readdir(joinpath(vnc_dir, poke)) if contains(x, ".rhd")]
    end
    idx = sortperm([minumum(rhd_times[x]) for x in poke_dirs])
    poke_dirs = poke_dirs[idx]



    # For each one:
    # - Get universal timestamp of start of first file
    # - Find matching Phy folder of spike sorted data
    # - Get which open-ephys experiments happened within this intan run 
end



# Function to load barcodes from intan files
function read_barcode_intan()

end
# Function to load barcodes from open-ephys 

# Function to load spike data from phy
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
# Function to load spike data from AMPS


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
    return time, data
end
