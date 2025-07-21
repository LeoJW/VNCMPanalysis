#-------- Functions for reading in estimation runs
function read_precision_kinematics_file!(df, file, task)
    precision_noise = h5read(joinpath(data_dir, file), "dict_0")
    precision_curves = h5read(joinpath(data_dir, file), "dict_1")
    subsets = h5read(joinpath(data_dir, file), "dict_2")
    mi_subsets = h5read(joinpath(data_dir, file), "dict_3")
    # Construct dataframe
    first_row = split(first(keys(precision_noise)), "_")
    names = vcat(first_row[1:2:end], ["mi", "precision", "precision_curve", "precision_noise", "subset", "mi_subset"])
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat(
        [x ? Float64 : String for x in is_numeric], 
        Float64, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}
    )
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(precision_noise)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        vals[findfirst(names .== "rep")] = task
        push!(thisdf, vcat(
            vals,
            precision_curves[key][1] .* log2(exp(1)), 
            find_precision_threshold(precision_noise[key] .* 1000, precision_curves[key][2:end]),
            [precision_curves[key] .* log2(exp(1))],
            [precision_noise[key] .* 1000],
            [subsets[key]],
            [mi_subsets[key] .* log2(exp(1))]
        ))
    end
    append!(df, thisdf)
end

function read_run_file!(df, file, task)
    precision_levels = h5read(joinpath(data_dir, file), "dict_0")
    precision_curves = h5read(joinpath(data_dir, file), "dict_1")
    precision_noise_curves = h5read(joinpath(data_dir, file), "dict_2")
    params = h5read(joinpath(data_dir, file), "dict_3")
    embed_mi = h5read(joinpath(data_dir, file), "dict_4")
    # Construct dataframe
    first_row = split(first(keys(precision_levels)), "_")
    names = vcat(first_row[1:2:end], ["mi", "precision", "precision_noise", "precision_curve", "precision_noise_curve", "precision_levels", "embed_mi"])
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat(
        [x ? Float64 : String for x in is_numeric], 
        Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}
    )
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(precision_levels)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        push!(thisdf, vcat(
            vals,
            precision_curves[key][1] .* log2(exp(1)), 
            find_precision_threshold(precision_levels[key] .* 1000, precision_curves[key][2:end]),
            find_precision_noise_threshold(precision_levels[key] .* 1000, precision_noise_curves[key]),
            [precision_curves[key] .* log2(exp(1))],
            [vec(mean(precision_noise_curves[key], dims=1)) .* log2(exp(1))],
            [precision_levels[key] .* 1000],
            [embed_mi[key]]
        ))
    end
    append!(df, thisdf)
end


#-------- Functions for reading in ancillary things

function get_neuron_statistics(; moths=moths, duration_thresh=10, fsamp=30000)
    thisdf = DataFrame()
    for moth in moths
        spikes = npzread(joinpath(data_dir, "..", moth * "_data.npz"))
        labels = npzread(joinpath(data_dir, "..", moth * "_labels.npz"))
        bouts = npzread(joinpath(data_dir, "..", moth * "_bouts.npz"))
        # Remove muscles
        for unit in keys(spikes)
            if (!).(occursin(r"[0-9]", unit))
                delete!(spikes, unit)
                delete!(labels, unit)
            end
        end
        # Get total flapping time 
        total_flapping_time = sum(bouts["ends"] .- bouts["starts"]) / fsamp
        # Pull out stats
        for neuron in keys(spikes)
            # Get firing rate in active periods, total span of time active
            diff_vec = diff(spikes[neuron])
            diff_over_thresh = findall(diff_vec .> (duration_thresh .* fsamp))
            end_inds = vcat(diff_over_thresh, length(spikes[neuron]))
            start_inds = vcat(1, diff_over_thresh .+ 1)
            spike_counts = end_inds - start_inds 
            bout_times = spikes[neuron][end_inds] - spikes[neuron][start_inds]
            
            mask = bout_times .!= 0
            mean_spike_rate = mean(spike_counts[mask] ./ bout_times[mask]) * fsamp
            total_time = sum(bout_times) / fsamp
            # Put as row in dataframe
            append!(thisdf, DataFrame(
                moth=moth,
                neuron=parse(Int, neuron),
                label=labels[neuron],
                nspikes=length(spikes[neuron]),
                meanrate=mean_spike_rate,
                timeactive=total_time,
                flapping_time=total_flapping_time
            ))
        end
    end
    return thisdf
end

function get_spikes(moth; refractory_thresh=1)
    moth_dir = joinpath(data_dir, "..", moth)

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
    return neurons, muscles, unit_details
end

# Get muscle statistics
# function read_muscle_spikes(; moths=moths)
#     for moth in moths
#         spikes = npzread(joinpath(data_dir, "..", moth * "_data.npz"))
#         if occursin(r"[0-9]", unit)
#             delete!(spikes, unit)
#         end

#     end   
# end

#-------- Varied utility functions

function find_precision_threshold(noise_levels, mi; threshold=0.9)
    # sg_window = 2 * floor(Int, length(noise_levels) / 5) + 1
    # sg_window = sg_window < 2 ? 5 : sg_window
    sg_window = 51
    curve = savitzky_golay(mi[2:end], sg_window, 2; deriv=0).y ./ mi[1]

    # curve = mi ./ mi[1]
    ind = findfirst(curve .< threshold)
    if isnothing(ind)
        return NaN
    else
        return noise_levels[ind]
    end
end
function find_precision_noise_threshold(noise_levels::Vector{Float64}, mi::Array{Float64,2}; threshold=0.9)
    # sg_window = 2 * floor(Int, length(noise_levels) / 5) + 1
    # sg_window = sg_window < 2 ? 5 : sg_window
    sg_window = 51
    meanmi = vec(mean(mi[:,2:end], dims=1))
    curve = savitzky_golay(meanmi, sg_window, 2; deriv=0).y ./ meanmi[1]

    ind = findfirst(curve .< threshold)
    if isnothing(ind)
        return NaN
    else
        return noise_levels[ind]
    end
end

function breakpoint(x, y; start_window=3)
    n_breaks = length(x) - 2 * start_window
    slope = zeros(n_breaks, 2)
    for (i,ind) in enumerate((start_window+1):(length(x)-start_window))
        mod1 = lm(@formula(y ~ x), DataFrame(x=x[1:ind], y=y[1:ind]))
        mod2 = lm(@formula(y ~ x), DataFrame(x=x[ind:end], y=y[ind:end]))
        slope[i,1] = coef(mod1)[2]
        slope[i,2] = coef(mod2)[2]
    end
    return slope
end
function find_scaling_point(x; threshold=0.5, allowed_above=2)
    boolvec = x .< threshold
    if !any(boolvec)
        return 1
    end
    vals, lens = rle(boolvec)
    # Find the first run where vals is false (meaning x >= threshold)
    # and the run length is greater than allowed_above
    cumulative_pos = 0
    for i in eachindex(vals)
        if !vals[i] && lens[i] >= allowed_above
            # Found the change point - return the position where this run starts
            return cumulative_pos + 1
        end
        cumulative_pos += lens[i]
    end
    # No change point found
    return length(x)
end