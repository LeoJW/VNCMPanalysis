using NPZ   # Read .npy files
using JSON  # Read json formatted files, duh
using DelimitedFiles
using Mmap
using BSplineKit
using DataFrames
using PooledArrays
using GLMakie
using BenchmarkTools
include("IntanReader.jl")
include("functions.jl")

moths = ["2023-05-20", "2023-05-25"]


vnc_dir = "/Volumes/PikesPeak/VNCMP"
motor_program_dir = "/Volumes/PikesPeak/VNCMP/MP_data/good_data"

# Constants everything should know
amps_muscle_order = [
    "lax", "lba", "lsa", "ldvm", "ldlm", 
    "rdlm", "rdvm", "rsa", "rba", "rax"]


# Time synchronize and match data from intan, open-ephys, and spike sorted sources


""" 
Function to match data for one individual moth
Takes two directory locations, parent dir of intan neural recordings and 
parent dir of motor program open-ephys recordings

Assumes parent dir of motor program recordings has spike sorting folder and Record Node folder 
"""
function read_and_match_moth_data(vnc_dir, mp_dir; verbose=false)
    voltage_threshold = 1.0 # V
    debounce_window = 30 # samples
    frame_dif_tol = 10 # samples
    check_n_events = 15 # samples
    fsamp = 30000 # Hz

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
        end_inds = [read_rhd_size(joinpath(vnc_dir, poke, x)) for x in rhd_files[poke]]
        rhd_start_ind[poke] = vcat(1, cumsum(end_inds[1:end-1]) .+ 1)
    end

    # Split out motor program folders
    mp_contents = readdir(mp_dir)
    amps_dir = joinpath(mp_dir, mp_contents[findfirst(occursin.("_spikesort", mp_contents))])
    openephys_dir = joinpath(mp_dir, mp_contents[findfirst(occursin.("Record Node", mp_contents))])
    # Match open-ephys experiments to each poke
    # Get open-ephys experiments, get start time in UTF
    # Assumes everything is in recording1
    experiments = [s for s in readdir(openephys_dir) if occursin("experiment", s)]
    oep_times = [readdlm(joinpath(openephys_dir, s, "recording1", "sync_messages.txt"))[1,10] for s in experiments]
    oep_times = trunc.(Int, oep_times / 1000) # ms to s
    
    # Load motor program spikes from AMPs
    amps_mat = get_amps_sort(amps_dir)
    
    # Create final spike data dataframe
    df = DataFrame(
        :moth => String[], 
        :poke => Int[], 
        :unit => String[], 
        :ismuscle => Bool[],
        :index => Int[], 
        :time => Float64[],
        :quality => String[], 
        :doublet => Bool[])
    neuron_names = []

    # For each experiment, use universal timestamps to guess which .rhd file in associated poke matches start time of experiment
    for (i, exp) in enumerate(experiments)
        # Find which rhd files happened after this experiment started
        time_compare = [findfirst(rhd_times[p] .> oep_times[i]) for p in poke_dirs]
        matching_poke_ind = findfirst((!).(isnothing.(time_compare)))
        experiment_poke = poke_dirs[matching_poke_ind]
        first_rhd = time_compare[matching_poke_ind] - 1
        # If open-ephys was started before intan index will be zero
        # Conditional here to catch those times
        if first_rhd == 0
            first_rhd += 1
        end
        #--- Grab motor program and neural spike event indices
        # Get all AMPS/motor program data for this experiment
        trial_start_times = readdlm(joinpath(amps_dir, "trial_start_times.txt"), ',', Any, '\n', header=true)[1]
        exp_num = [parse(Int, split(x, "_")[2][end]) for x in trial_start_times[:,1]] .+ 1
        trial_start_ind = Dict{Int, Int}(parse(Int, split(x[1], "_")[3][1:3]) => x[2] for x in eachrow(trial_start_times))
        trials_in_this_experiment = findall(exp_num .== parse(Int, exp[end]))
        amps_trial_inds = vec(any(amps_mat[:,1] .== trials_in_this_experiment', dims=2))
        mask = amps_trial_inds .&& (amps_mat[:,6] .== 1)
        mp_spike_inds = Int.(amps_mat[mask,4])
        # Shift spike indices by starting index of each trial, +1 because moving from 0-index python to 1-index julia
        mp_spike_inds = mp_spike_inds .+ [trial_start_ind[x] for x in amps_mat[mask,1]] .+ 1
        # for (i_muscle, muscle) in enumerate(amps_muscle_order)
        #     append!(spikedata[muscle], mp_spike_inds[amps_mat[mask,2] .== (i_muscle - 1)])
        # end
        # Add motor program data to final dataframe
        df = vcat(df, DataFrame(
            :moth => splitpath(vnc_dir)[end],
            :poke => parse(Int, split(experiment_poke, "_")[1][end]),
            :unit => [amps_muscle_order[Int(x+1)] for x in amps_mat[mask,2]],
            :index => mp_spike_inds,
            :time => mp_spike_inds ./ fsamp,
            :ismuscle => true,
            :quality => "good",
            :doublet => false
        ))
        
        #---- Sample alignment for intan data
        # Load run of .rhd files and open-ephys data where we have AMPS spikes
        # Preallocate the three digital channels for open-ephys and intan
        # I use this fancy grow-size method for intan, but oep I just set once
        initial_capacity = 50000
        digital_names = ["barcode", "requestsend", "frame", "trigger"]
        rhd_digital = Dict{String, Vector{Int}}(key => Vector{Int}(undef, initial_capacity) for key in digital_names)
        oep_digital = Dict{String, Vector{Int}}(key => Vector{Int}(undef, initial_capacity) for key in digital_names)
        ind_rhd = Dict{String, Int}(key => 1 for key in digital_names)
        # Load each .rhd file, get indices of flip times
        # Just uses low-high transitions. I assume this is enough information to resolve sync
        # Note camera frame signal is normal high, then drops low during exposure. Frame times will be 
        rhdtime = Vector{Float64}([])
        bob = Vector{Float64}([])
        for (index_rhd, rhd_file) in enumerate(rhd_files[experiment_poke])
            # TODO: Switch back to not using rhd time when I'm done debugging
            dat = read_data_rhd(joinpath(vnc_dir, experiment_poke, rhd_file), read_amplifier=false, read_adc=true)
            adc = dat["adc"]
            # adc = read_data_rhd(joinpath(vnc_dir, experiment_poke, rhd_file), read_amplifier=false, read_adc=true)["adc"]
            shift_indices = rhd_start_ind[experiment_poke][index_rhd] - 1
            append!(rhdtime, dat["time"])
            append!(bob, adc[4,:])
            # Loop over each digital channel, get indices of low-high transitions
            for (j, channel) in enumerate(digital_names)
                crossing_inds = find_threshold_crossings(adc[j,:], voltage_threshold, debounce_window) .+ shift_indices
                # Check if more capacity is needed before adding elements, double capacity if needed
                if (ind_rhd[channel] + length(crossing_inds)) > length(rhd_digital[channel])
                    resize!(rhd_digital[channel], 2 * length(rhd_digital[channel]))
                end
                rhd_digital[channel][ind_rhd[channel]:ind_rhd[channel]+length(crossing_inds)-1] = crossing_inds
                ind_rhd[channel] += length(crossing_inds)
                # TODO: While you're here, actually translate barcodes, save start index and value for each
            end
        end
        # Trim digital channels to final size
        for channel in digital_names
            resize!(rhd_digital[channel], ind_rhd[channel] - 1)
        end
        # Load open-ephys data for experiment, get indices of low-high transitions
        oeptime, oep_data = read_binary_open_ephys(joinpath(openephys_dir, exp, "recording1"), [19,20,21,22])
        for (j, channel) in enumerate(digital_names)
            oep_digital[channel] = find_threshold_crossings(oep_data[:,j], voltage_threshold, debounce_window)
        end

        # Map all event samples between intan and oep
        # Initial mapping from trigger. Requires that we find exact match for each trigger pulse
        if length(rhd_digital["trigger"]) != length(oep_digital["trigger"])
            error("Intan DAQ has $(length(rhd_digital["trigger"])) trigger presses while open-ephys has $(length(oep_digital["trigger"]))!")
        end
        oep_events, rhd_events = oep_digital["trigger"], rhd_digital["trigger"]
        # Add camera frame events
        # Start at first frame after first trigger, match working out
        oep_first = findfirst(oep_digital["frame"] .> oep_digital["trigger"][1])
        rhd_first = findfirst(rhd_digital["frame"] .> rhd_digital["trigger"][1])
        append!(oep_events, oep_digital["frame"][oep_first])
        append!(rhd_events, rhd_digital["frame"][rhd_first])
        match_events!(oep_events, rhd_events, oep_digital["frame"], rhd_digital["frame"], oep_first, rhd_first;
            frame_dif_tol=frame_dif_tol, check_n_events=check_n_events)
        # TODO: Add other event channels if needed
        # BSpline interpolation to get function to convert from intan samples to oep 
        # open-ephys is treated as master because it actually ran at 30kHz 
        sort_idx = sortperm(oep_events)
        oep_events, rhd_events = oep_events[sort_idx], rhd_events[sort_idx]
        itp = interpolate(rhd_events, oep_events, BSplineOrder(4))

        # Now that we have alignment, load neural spikes from Phy matching this poke/experiment
        neurons, unit_details, sort_params = read_phy_spikes(joinpath(vnc_dir, "phy_folder", phy_dirs[matching_poke_ind]))
        # Catch maximum unit number of last poke's units
        prev_max_unit_number = length(neuron_names) > 0 ? maximum(neuron_names) : 0
        # Add each phy unit to data
        for key in keys(neurons)
            # Assign unit names/numbers that are unique across multiple pokes of same moth
            unit_number = !(key in neuron_names) ? key : prev_max_unit_number + key + 1
            append!(neuron_names, unit_number)
            indices = round.(Int, itp.(neurons[key]))
            df = vcat(df, DataFrame(
                :moth => splitpath(vnc_dir)[end],
                :poke => parse(Int, split(experiment_poke, "_")[1][end]),
                :unit => string(unit_number),
                :index => indices,
                :time => indices ./ fsamp,
                :ismuscle => false,
                :quality => unit_details["quality"][key],
                :doublet => unit_details["doublet"][key]
            ))
        end
    end
    # NOTE: Unsure if better to pool before or after full dataframe is built. 
    # For now pooling after as it feels like the safest move. 
    # But you CAN concatenate pooled arrays, I'm just not sure if that break something
    df.moth = PooledArray(df.moth, compress=true)
    df.poke = PooledArray(df.poke, compress=true)
    df.unit = PooledArray(df.unit, compress=true)
    df.quality = PooledArray(df.quality, compress=true)
    df.doublet = PooledArray(df.doublet, compress=true)

    return df
end

df = vcat(
    read_and_match_moth_data(
        "/Volumes/PikesPeak/VNCMP/2023-05-25", 
        "/Volumes/PikesPeak/VNCMP/MP_data/good_data/2023-05-25_12-24-05"),
    read_and_match_moth_data(
        "/Volumes/PikesPeak/VNCMP/2023-05-20", 
        "/Volumes/PikesPeak/VNCMP/MP_data/good_data/2023-05-20_15-36-35")
)


##
# mat = read_data_rhd(path, read_amplifier=false, read_adc=true)["adc"]
f = Figure()
ax = Axis(f[1,1])
for i in 22
    # lines!(ax, dat[1:100000,i], label=string(i))
    lines!(ax, dat[:,i], label=string(i))
end
f[1,2] = Legend(f, ax)
display(f)

## More drift than you might think

f = Figure()
ax = Axis(f[1,1])
sub = 5
alignto = 1
shift = rhd_digital["trigger"][alignto] - oep_digital["trigger"][alignto]
y = oep_data[1:sub:end,3]
lines!(ax, oeptime[1:sub:end] .- oeptime[oep_digital["trigger"][alignto]], [y[1:end-1]..., NaN])
# scatter!(ax, oep_digital["trigger"][alignto], 0, markersize=20)
scatter!(ax, 0, 0, markersize=20)
why = bob[1:sub:end]
lines!(ax, rhdtime[1:sub:end] .- rhdtime[rhd_digital["trigger"][alignto]], [why[1:end-1]..., NaN])
# scatter!(ax, rhd_digital["trigger"][alignto] - shift, 0, markersize=10)
display(f)

##

itp = interpolate(rhd_events, oep_events, BSplineOrder(4))
xitp = extrema(rhd_events)[1]:extrema(rhd_events)[2]

f = Figure()
ax = Axis(f[1,1])
scatter!(ax, rhd_events .- rhd_events[1], oep_events .- oep_events[1])
lines!(ax, xitp .- rhd_events[1], itp.(xitp) .- oep_events[1])
display(f)

##
f = Figure()
ax = Axis(f[1,1])
scatter!(ax, spikedata["ldlm"] ./ 30000, zeros(length(spikedata["ldlm"])))
scatter!(ax, spikedata["rdlm"] ./ 30000, zeros(length(spikedata["rdlm"])))
current_figure()


## A fitting approach
oep = oep_digital["trigger"] .- oep_digital["trigger"][1]
rhd = rhd_digital["trigger"] .- rhd_digital["trigger"][1]
# fit(oep, rhd)
scatter(oep, rhd)
ablines!([0], [1])
current_figure()

## How long for how much drift
alignto = 1
shift = rhd_digital["trigger"][alignto] - oep_digital["trigger"][alignto]
drift = (oep_digital["trigger"] .- (rhd_digital["trigger"] .- shift) ) ./ 30000
scatter((oep_digital["trigger"] .- oep_digital["trigger"][alignto]) / 30000, drift)

# NI DAQ seems to run almost exactly 1ms faster per every second of runtime 
# This is b/c intan DAQ actually ran at 29999.999666470107 Hz
