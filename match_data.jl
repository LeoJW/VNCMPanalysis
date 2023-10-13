using NPZ   # Read .npy files
using JSON  # Read json formatted files, duh
using DelimitedFiles
using Mmap
using GLMakie
include("IntanReader.jl")
include("functions.jl")

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
    voltage_threshold = 1.0 # V

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
    # Then determine starting index of each rhd file in each poke
    rhd_files = Dict{String, Vector{String}}()
    rhd_times = Dict{String, Vector{Float64}}()
    rhd_start_ind = Dict{String, Vector{Int}}()
    for poke in poke_dirs
        rhd_files[poke] = [x for x in readdir(joinpath(vnc_dir, poke)) if contains(x, ".rhd")]
        rhd_times[poke] = [ctime(joinpath(vnc_dir, poke, x)) for x in rhd_files[poke]]
    end
    poke_dirs = poke_dirs[sortperm([minimum(rhd_times[x]) for x in poke_dirs])]
    for poke in poke_dirs
        rhd_start_ind[poke] = [read_rhd_size(joinpath(vnc_dir, poke, x)) for x in rhd_files[poke]]
        # Cumulative sum... but check the indexing!
    end

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
        experiment_poke = poke_dirs[poke_match_ind]
        rhd_match_ind = time_compare[poke_match_ind] - 1
        # If open-ephys was started before intan index will be zero
        # Conditional here so I can note those times
        if rhd_match_ind == 0
            rhd_match_ind += 1
        end
        # Get all AMPS data for this experiment
        trial_start_times = readdlm(joinpath(amps_dir, "trial_start_times.txt"), ',', Any, '\n', header=true)[1]
        exp_num = [parse(Int, split(x, "_")[2][end]) for x in trial_start_times[:,1]] .+ 1
        trial_start_ind = Dict{Int, Int}(parse(Int, split(x[1], "_")[3][1:3]) => x[2] for x in eachrow(trial_start_times))
        trials_in_this_experiment = findall(exp_num .== parse(Int, exp[end]))
        amps_trial_inds = vec(any(amps_mat[:,1] .== trials_in_this_experiment', dims=2))
        mask = amps_trial_inds .&& (amps_mat[:,6] .== 1)
        mp_spike_inds = Int.(amps_mat[mask,4])
        # Shift spike indices by starting index of each trial, +1 because moving from 0-index python to 1-index julia
        mp_spike_inds = mp_spike_inds .+ [trial_start_ind[x] for x in amps_mat[mask,1]] .+ 1
        # Preallocate the three digital channels for open-ephys and intan
        initial_capacity = 10000
        rhd_digital = Dict{String, Vector{Int}}(
            "barcode" => Vector{Int}(undef, initial_capacity),
            "requestsend" => Vector{Int}(undef, initial_capacity),
            "frames" => Vector{Int}(undef, initial_capacity)
        )
        oep_digital = Dict{String, Vector{Int}}(
            "barcode" => Vector{Int}(undef, initial_capacity),
            "requestsend" => Vector{Int}(undef, initial_capacity),
            "frames" => Vector{Int}(undef, initial_capacity)
        )
        i_rhd = [1,1,1] # barcode, rts, frames
        i_oep = [1,1,1] # barcode, rts, frames
        # Load run of .rhd files and open-ephys data where we have AMPS spikes
        # Load each .rhd file in run starting from rhd_match_ind, get indices of flip times
        for rhd_file in rhd_files[experiment_poke]
            adc = read_data_rhd(joinpath(vnc_dir, experiment_poke, rhd_file), read_amplifier=false, read_adc=true)["adc"]
            # rhd_digital["barcode"] = findall(adc[1,:] .> voltage_threshold) .+ 
            rhd_digital["requestsend"] = findall(adc[2,:] .> voltage_threshold)
            rhd_digital["frames"] = findall(adc[3,:] .> voltage_threshold)
        end
        # Load open-ephys data for experiment, get indices of flip times

        # Come up with sample mapping based on barcodes
        # Look for initial phase shift, in samples

        # Do that correction, then look for further distortions


        
        # return trial_start_times, mp_spike_inds
    end
end

rhd_files, rhd_times = read_and_match_moth_data(
    "/Volumes/PikesPeak/VNCMP/2023-05-25", 
    "/Volumes/PikesPeak/VNCMP/MP_data/good_data/2023-05-25_12-24-05")

##
mat = read_data_rhd(path, read_amplifier=false, read_adc=true)["adc"]
f = Figure()
ax = Axis(f[1,1])
for i in eachrow(mat)
    lines!(ax, i)
end
display(f)
