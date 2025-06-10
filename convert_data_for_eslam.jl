using DelimitedFiles
using NPZ
using JSON
using DataFrames
using DataFramesMeta
using DataStructures
using GLMakie

include("functions.jl")

##

moths = [
    "2025-02-25",
    "2025-02-25_1",
    "2025-03-11",
    "2025-03-12_1",
    "2025-03-20",
    "2025-03-21"
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
    bout_starts = min.(muscles["ldlm"][start_inds[valid_periods]] .- (buffer_in_sec * fsamp), 1)
    bout_ends = muscles["ldlm"][end_inds[valid_periods]] .+ (buffer_in_sec * fsamp)
    # Clear out spikes in neurons and muscles not valid by bouts
    for unit in keys(neurons)
        valid = zeros(Bool, length(neurons[unit]))
        for i in eachindex(bout_starts)
            valid[(neurons[unit] .> bout_starts[i]) .&& (neurons[unit] .< bout_ends[i])] .= true
        end
        keepat!(neurons[unit], valid)
    end
    for unit in keys(muscles)
        valid = zeros(Bool, length(muscles[unit]))
        for i in eachindex(bout_starts)
            valid[(muscles[unit] .> bout_starts[i]) .&& (muscles[unit] .< bout_ends[i])] .= true
        end
        keepat!(muscles[unit], valid)
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
    npzwrite(joinpath(data_dir, moth * "_bouts.npy"), Dict("starts" => bout_starts, "ends" => bout_ends))
end


##

moth = moths[4]
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
seg = 1 / (length(mua_units)+1)
for (i,unit) in enumerate(mua_units)
    vlines!(ax, neurons[unit] ./ fsamp, ymin=i*seg, ymax=(i+1)*seg)
end
vlines!(ax, muscles["ldlm"] ./ fsamp, ymin=0.0, ymax=seg)
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
