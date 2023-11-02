using Base.Threads: nthreads, @threads, @spawn
using Base.Iterators: partition # Multithreading
using NPZ   # Read .npy files
using JSON  # Read json formatted files, duh
using JLD
using DelimitedFiles
using Mmap
using BSplineKit
using DataFrames
using DataFramesMeta
using PooledArrays
using Pipe
using GLMakie
using AlgebraOfGraphics
using CausalityTools
using BenchmarkTools
include("IntanReader.jl")
include("functions.jl")
include("precision_functions.jl")

moths = ["2023-05-20", "2023-05-25"]


vnc_dir = "/Volumes/PikesPeak/VNCMP"
motor_program_dir = "/Volumes/PikesPeak/VNCMP/MP_data/good_data"
analysis_dir = @__DIR__

# Constants everything should know (effectively global)
# NOTE: I suspect LAX and LBA got flipped in split_motor_program.py, so flipping them back here. But true AMPS order is lax, lba
muscle_order = [
    "lba", "lax", "lsa", "ldvm", "ldlm", 
    "rdlm", "rdvm", "rsa", "rba", "rax"]
phase_wrap_thresholds = Dict("ax"=>0.32, "ba"=>0.6, "sa"=>0.4, "dvm"=>3.0, "dlm"=>3.0)


##
""" 
Function to match data for one individual moth
Takes two directory locations, parent dir of intan neural recordings and 
parent dir of motor program open-ephys recordings

Assumes parent dir of motor program recordings has spike sorting folder and Record Node folder 
"""
function read_and_match_moth_data(vnc_dir, mp_dir; 
    wingbeat_muscle="ldlm",
    exclusion_period_ms=1.0)
    # Important constants not worth making input arguments
    voltage_threshold = 1.0 # V
    debounce_window = 30 # samples
    frame_dif_tol = 10 # samples
    check_n_events = 15 # samples
    fsamp = 30000 # Hz
    wingbeat_diff_threshold = 1000 # samples, 33ms at 30kHz

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
    
    # Initialize final output dataframe
    df = DataFrame(
        :moth => String[], 
        :poke => Int[], 
        :unit => String[], 
        :ismuscle => Bool[],
        :index => Int[], 
        :abstime => Float64[], # Spike time relative to recording start
        :time => Float64[], # Spike time relative to wingbeat start
        :quality => String[], 
        :doublet => Bool[],
        :wb => Int[],
        :wblen => Float64[])
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
        # Make motor program dataframe
        local dfmp = DataFrame(
            :moth => splitpath(vnc_dir)[end],
            :poke => parse(Int, split(experiment_poke, "_")[1][end]),
            :unit => [muscle_order[Int(x+1)] for x in amps_mat[mask,2]],
            :index => mp_spike_inds,
            :abstime => mp_spike_inds ./ fsamp,
            :time => 0.0,
            :ismuscle => true,
            :quality => "good",
            :doublet => false,
            :wb => 0,
            :wblen => 0
        )

        #--- Create wingbeats
        # Determine wingbeat start and end indices as "bins"
        wb_muscle_inds = dfmp[dfmp.unit .== wingbeat_muscle, :index]
        wb_muscle_diff = diff(wb_muscle_inds)
        wb_starts = wb_muscle_inds[findall(wb_muscle_diff .>= wingbeat_diff_threshold) .+ 1]
        # Assign each spike to a wingbeat
        dfmp.wb = searchsortedlast.(Ref(wb_starts), dfmp.index)
        # Make wingbeat period/frequency column
        # Set first to NaN for cleaning later (relative times unknowable for first wingbeat)
        # Set last to previous length (approx close enough)
        wblen = diff(wb_starts)
        wblen = vcat(NaN, wblen, wblen[end])
        dfmp.wblen = wblen[dfmp.wb .+ 1] # +1 as first wingbeat will be counted as zero
        # Calculate spike time relative to wingbeat start, phase
        dfmp.time = (dfmp.index .- vcat(0, wb_starts)[dfmp.wb .+ 1]) ./ fsamp
        # Make wingstroke numbers unique based on max in overall dataframe
        max_wb_number = length(df.wb) > 0 ? maximum(df.wb) : 0
        dfmp.wb .+= max_wb_number
        # Add local motor program dataframe to full
        df = vcat(df, dfmp)
        
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
        for (index_rhd, rhd_file) in enumerate(rhd_files[experiment_poke])
            adc = read_data_rhd(joinpath(vnc_dir, experiment_poke, rhd_file), read_amplifier=false, read_adc=true)["adc"]
            shift_indices = rhd_start_ind[experiment_poke][index_rhd] - 1
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
        _, oep_data = read_binary_open_ephys(joinpath(openephys_dir, exp, "recording1"), [19,20,21,22])
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
        # Remove duplicated spikes within time radius
        remove_duplicate_spikes!(neurons; exclusion_period_ms=exclusion_period_ms, fsamp=fsamp)
        # Catch maximum unit number of last poke's units
        prev_max_unit_number = length(neuron_names) > 0 ? maximum(neuron_names) : 0
        # Add each phy unit to data
        for key in keys(neurons)
            # Assign unit names/numbers that are unique across multiple pokes of same moth
            unit_number = !(key in neuron_names) ? key : prev_max_unit_number + key + 1
            append!(neuron_names, unit_number)
            indices = round.(Int, itp.(neurons[key]))
            dtemp = DataFrame(
                :moth => splitpath(vnc_dir)[end],
                :poke => parse(Int, split(experiment_poke, "_")[1][end]),
                :unit => string(unit_number),
                :index => indices,
                :abstime => indices ./ fsamp,
                :time => 0.0,
                :ismuscle => false,
                :quality => unit_details["quality"][key],
                :doublet => unit_details["doublet"][key],
                :wb => searchsortedlast.(Ref(wb_starts), indices),
                :wblen => 0
            )
            dtemp.wblen = wblen[dtemp.wb .+ 1]
            dtemp.time = (indices .- vcat(0, wb_starts)[dtemp.wb .+ 1]) ./ fsamp
            dtemp.wb .+= max_wb_number
            df = vcat(df, dtemp)
        end
    end
    # Remove data from 0th wingbeats, marked as wblen of NaN
    df = @subset(df, (!).(isnan.(:wblen)))
    # Rescale wblen to units of seconds
    df.wblen ./= fsamp

    # NOTE: Unsure if better to pool before or after full dataframe is built. 
    # For now pooling after as it feels like the safest move. 
    # You CAN concatenate pooled arrays, I'm just not sure if that breaks something
    for name in [col for col in names(df) if col ∉ ["index", "abstime", "time", "wb", "wblen"]]
        df[:, name] = PooledArray(df[:, name], compress=true)
    end

    # TODO: Some way to return visual stimuli, either as its own data structure or in the dataframe
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

# Post-processing
df = @pipe df |> 
    @transform(_, :wbfreq = 1 ./ :wblen) |> 
    # Clean out wingbeats below a frequency threshold
    @subset(_, :wbfreq .> 10) |> 
    # rdlm barely has any data so go ahead and remove
    @subset(_, :unit .!= "rdlm") |> 
    # Remove neural units that are noise
    @subset(_, :quality .!= "noise") |> 
    # Final wingbeats have neural spikes that drag on for many seconds
    # May be useful later, but for now removing any neural spikes past when wingbeat should end
    @subset(_, :time .<= :wblen) |> 
    # Make phase column
    @transform(_, :phase = :time ./ :wblen) |> 
    # Unwrap muscle spikes
    groupby(_, [:moth, :poke]) |> 
    transform!(_, [:time, :phase, :wb, :wblen, :unit, :ismuscle] => unwrap_spikes_to_prev => [:time, :phase, :wb, :wblen]) |> 
    @subset(_, (!).(isnan.(:time)))

## Settings for all plots

dark_or_light = "light"
if dark_or_light .== "light"
    set_theme!()
    figsdir = joinpath(analysis_dir, "figs_light")
else
    set_theme!(theme_dark())
    figsdir = joinpath(analysis_dir, "figs_dark")
end
update_theme!(fontsize=30)


## Histograms of muscle spike phase

@pipe df |> 
    @subset(_, :ismuscle) |> 
    @subset(_, :phase .!= 0.0) |> 
    (
    AlgebraOfGraphics.data(_) *
    mapping(:phase, color=:moth, row=:unit) *
    histogram(bins=100, normalization=:pdf, datalimits=extrema) *
    visual(alpha=0.6)
    ) |> 
    draw(_, 
        figure=(resolution=(1000, 1800),),
        facet=(; linkxaxes=:colwise, linkyaxes=:none),
        axis=(; limits=(nothing, nothing))) #|> 
    # save(joinpath(figsdir, "muscle_phase_hist_with_1centered_unwrap.png"), _)
current_figure()

## Dimensionality checking. 
# Max number of spikes per wingbeat combinatorially between muscles and neurons?
max_n_spike_per_wb = @pipe df |> 
    groupby(_, [:moth, :poke, :unit, :wb]) |> 
    combine(_, nrow) |> 
    groupby(_, [:moth, :poke, :unit]) |> 
    combine(_, :nrow => maximum)

# How many rows (wingstrokes) vs columns (max num spikes) for each combination of (neuron, muscle)
nwb_combos = Dict{Tuple, Int}()
for gdf in groupby(df, [:moth, :poke, :wb])
    muscles = unique(gdf[gdf.ismuscle, :unit])
    neurons = unique(gdf[(!).(gdf.ismuscle), :unit])
    for combos in Base.product(muscles, neurons)
        key = (first(gdf.moth), first(gdf.poke), combos[1], combos[2])
        if combos[1] == combos[2]
            continue
        elseif !haskey(nwb_combos, key)
            nwb_combos[key] = 1
        else
            nwb_combos[key] += 1
        end
    end
end

# For each combination get overall dimensionality
num_wingbeats = Int[]
num_dimensions = Int[]
num_dim_firstmuscle = Int[]
for key in keys(nwb_combos)
    append!(num_wingbeats, nwb_combos[key])
    dim1 = @subset(max_n_spike_per_wb, (:moth .== key[1]) .&& (:poke .== key[2]) .&& (:unit .== key[3])).nrow_maximum[1]
    dim2 = @subset(max_n_spike_per_wb, (:moth .== key[1]) .&& (:poke .== key[2]) .&& (:unit .== key[4])).nrow_maximum[1]
    append!(num_dimensions, dim1 + dim2)
    append!(num_dim_firstmuscle, 1 + dim2)
end

f = Figure(resolution=(1200, 900))
ax = Axis(f[1,1],
    xlabel="Number of wingbeats for (muscle, neuron) pairs",
    ylabel="Overall dimensionality (# of columns in input + output)")
scatter!(ax, num_wingbeats, num_dimensions)
save(joinpath(figsdir, "dimensionality_vs_nwingbeats.png"), f)
current_figure()

f = Figure(resolution=(1200, 900))
ax = Axis(f[1,1],
    xlabel="Number of wingbeats for (muscle, neuron) pairs",
    ylabel="Overall dimensionality (# of columns in input + output)")
scatter!(ax, num_wingbeats, num_dimensions, label="Original")
scatter!(ax, num_wingbeats, num_dim_firstmuscle, label="Muscles only first spike")
axislegend()
save(joinpath(figsdir, "dimensionality_vs_nwingbeats_muscles_only_one.png"), f)
current_figure()


##--- Calculate and plot precision for each (muscle, neuron) combination
# Create list of all muscle, unit combinations for each moth
combinations = Dict{String, Set}(moth => Set([]) for moth in unique(df.moth))
for gdf in groupby(df, [:moth, :poke, :wb])
    muscles = unique(gdf[gdf.ismuscle, :unit])
    neurons = unique(gdf[(!).(gdf.ismuscle), :unit])
    for combos in Base.product(muscles, neurons)
        if combos[1] == combos[2]
            continue
        elseif !(combos in combinations[gdf.moth[1]])
            push!(combinations[gdf.moth[1]], combos)
        end
    end
end
# group by moth, poke, loop over (muscle, neuron) combinations, calculate precision and make a plot
prec = Dict{Tuple{String, String, String}, Float64}()
for gdf in groupby(df, :moth)
    thismoth = gdf.moth[1]
    for combo in combinations[thismoth]
        X, Y = XY_array_from_dataframe((combo[1], combo[2]), gdf.unit, gdf.time, gdf.wb)
        if size(X,1) <= (size(X,2) + size(Y,2))
            prec[(thismoth, combo[1], combo[2])] = 0.0
            continue
        end
        println(X)
        f = Figure()
        ax = Axis(f[1,1], xscale=log10, xlabel="Added noise amplitude (ms)", ylabel="MI (bits)")
        precision_val = precision(X, Y;
            noise=exp10.(range(log10(0.05), stop=log10(10), length=100)),
            repeats=100,
            do_plot=true,
            ax=ax
        )
        prec[(thismoth, combo[1], combo[2])] = precision_val
        ax.title = "Moth $thismoth, Muscle: $(combo[1]), Neuron: $(combo[2])"
        save(joinpath(figsdir, "precision_curves", join(thismoth, combo[1], combo[2], "_") * ".png"), f)
    end
end

##

gdf = first(groupby(df, :moth))
combo = first(combinations[gdf.moth[1]])
X, Y = XY_array_from_dataframe((combo[1], combo[2]), gdf.unit, gdf.time, gdf.wb)

nspike = vec(sum((!).(isnan.(X)), dims=2))
[sum(nspike .== n) for n in unique(nspike)]

# TODO: Better system that saves parameters I ran this under (like number of repeats)
save(joinpath(analysis_dir, "precision_rough_first_pass.jld"), "GOV", prec)


## Plot of precision curves, facet per muscle


## Plot all units against time for a couple wingstrokes

dt = @subset(df, (:moth .== "2023-05-25") .&& (:wb .>= 1000) .&& (:wb .<= 1004))

f = Figure(resolution=(1800, 1000))
ax = Axis(f[1,1], xlabel="Time (s)")
vlines!(dt[dt.unit .== "ldlm" .&& dt.phase .== 1.0, :abstime];
    color=:grey)
increment = 1 / (length(unique(dt.unit)) + 2)
value_to_index = Dict(key => i for (i, key) in enumerate(unique(dt.wb)))
for (i, (key, gdf)) in enumerate(pairs(groupby(dt, :unit)))
    wbvec = [value_to_index[value] for value in gdf.wb]
    vlines!(ax, gdf.abstime;
        ymin=i * increment - increment * 0.4,
        ymax=i * increment + increment * 0.4,
        linewidth = gdf.ismuscle[1] ? 4 : 1.5,
        # linestyle = gdf.ismuscle[1] ? :dashdot : :solid,
        label=key[1],
        color=wbvec,
        colormap=:tab10,
        colorrange=(1,10))
end
hideydecorations!(ax)
save(joinpath(figsdir, "example_spikes_across_several_wb_1centered.png"), f)
current_figure()

## group by unit, look at ISI to catch overlapping spikes

for (key, gdf) in pairs(groupby(@subset(df, (!).(:ismuscle)), [:moth, :poke, :unit]))
    diftime = diff(gdf.abstime)
    diftime = diftime[diftime .< 0.2]
    if length(diftime) < 1
        continue
    end
    f = Figure()
    ax = Axis(f[1,1])
    hist!(ax, diftime; bins=100, normalization=:pdf)
    save(joinpath(figsdir, "ISI", key[1] * "_" * string(key[3]) * ".png"), f)
end

## Wingbeat frequency plot

@pipe df |> 
    (
    AlgebraOfGraphics.data(_) *
    mapping(:wbfreq => "Wingbeat Frequency (Hz)", color=:moth => "Moth") * 
    histogram(bins=100, normalization=:pdf, datalimits=extrema) *
    visual(alpha=0.5)
    ) |> 
    draw(_,
        figure=(resolution=(1600, 800),)) |> 
    save(joinpath(figsdir, "wingbeat_frequency.png"), _)
current_figure()

##

bob = @pipe df |> 
    groupby(_, [:moth, :poke, :unit]) |> 
    combine(_, nrow)

jim = @pipe df |> 
    @subset(_, (!).(:ismuscle)) |> 
    groupby(_, [:moth, :poke, :unit, :quality]) |> 
    combine(_, nrow)
