"""
spike_MI

The basic, vanilla-flavored calculation of MI between arrays X and Y
X is spike times or phases, each row is a wingbeat
Y is some metric of behavioral output, often PCs of yaw torque
"""
function spike_MI(x::Array{Float64}, y::Array{Float64}; estimator=GaoOhViswanath(k=4))
    # Add extremely small noise (somehow brings information estimate to correct value for some data)
    x = x .+ (1e-7 * rand(size(x, 1), size(x, 2)))
    # How many spikes per wingbeat from X
    nspike = vec(sum((!).(isnan.(x)), dims=2))
    # Probability of each possible number of spikes (except zero)
    unq = unique(nspike)
    unq = unq[unq .!= 0]
    probs = zeros(maximum(unq))
    probs[unq] .= [sum(nspike .== u) / length(nspike) for u in unq]
    # Loop over number of spikes per wb
    mi = 0
    for sp in unq
        mask = nspike .== sp
        nwb = sum(mask)
        if (nwb > sp) && (estimator.k < nwb)
            mi += mutualinfo(estimator, StateSpaceSet(x[mask, 1:sp]), StateSpaceSet(y[mask, :])) * probs[sp]
        end
    end
    return mi
end


"""
repeat_corrupted_MI
x - Array of spike times/phases, each row is a wingbeat or behavioral separator. NaNs expected for fill
y - Array of output variables, likely PCs of body torque or force, each row is a wingbeat

MI - Array, rows are noise levels and columns are repeats at each noise level
"""
function repeat_corrupted_MI(x::Array{Float64}, y::Array{Float64}; 
    estimator=GaoOhViswanath(k=4), 
    repeats::Int=150, 
    noise::Vector{Float64}=exp10.(range(log10(0.05), stop=log10(6), length=120)))
    # Determine how many spikes per wingbeat, probability of each possible number of spikes (except zero)
    nspikeX, nspikeY = vec(sum((!).(isnan.(x)), dims=2)), vec(sum((!).(isnan.(y)), dims=2))
    nspike_combinations = unique(eachrow(hcat(nspikeX, nspikeY)))
    probs = Dict(combo => sum((nspikeX .== combo[1]) .&& (nspikeY .== combo[2])) / length(nspikeX) for combo in nspike_combinations)

    # Partition noise levels into chunks that tasks will deal with
    tasks_per_thread = 1 # customize this as needed. More tasks have more overhead, but better load balancing
    chunk_size = max(1, length(noise) ÷ (tasks_per_thread * nthreads()))
    data_chunks = partition(noise, chunk_size) 

    tasks = map(data_chunks) do chunk
        @spawn begin
            local_mi = zeros(length(chunk), repeats)
            Yn = zeros(size(y))
            for (i,noiselvl) in enumerate(chunk)
                # Loop over repeats
                for j in 1:repeats
                    # Loop over number of spikes per wb
                    mi = 0
                    for combo in nspike_combinations
                        mask = (nspikeX .== combo[1]) .&& (nspikeY .== combo[2])
                        nwb = sum(mask)
                        @inbounds if (nwb > sum(combo)) && (estimator.k < nwb)
                            Yn = y[mask, 1:combo[2]] .+ (noiselvl * rand(nwb, combo[2]))
                            X = StateSpaceSet(x[mask, 1:combo[1]])
                            Y = StateSpaceSet(Yn)
                            mi += mutualinfo(estimator, X, Y) * probs[combo]
                        end
                    end
                    local_mi[i,j] = mi
                end
            end
            return local_mi
        end
    end
    # Collect results from each thread
    states = fetch.(tasks) 
    # Assmeble vector of vectors into matrix
    MI = reduce(vcat, states)
    return MI
end
# Version at single noise level
function repeat_corrupted_MI(x::Array{Float64}, y::Array{Float64}, noise::Float64; 
    estimator=GaoOhViswanath(k=4), 
    repeats::Int=150)
    # How many spikes per wingbeat from X
    nspike = vec(sum((!).(isnan.(x)), dims=2))
    # Probability of each possible number of spikes (except zero)
    unq = unique(nspike)
    unq = unq[unq .!= 0]
    probs = zeros(maximum(unq))
    probs[unq] .= [sum(nspike .== u) / length(nspike) for u in unq]
    # Partition repeats into chunks that tasks will deal with
    tasks_per_thread = 1 # customize this as needed. More tasks have more overhead, but better load balancing
    chunk_size = max(1, repeats ÷ (tasks_per_thread * nthreads()))
    data_chunks = partition(1:repeats, chunk_size) 
    tasks = map(data_chunks) do chunk
        @spawn begin
            local_mi = zeros(length(chunk))
            # Loop over repeats
            for (i,_) in enumerate(chunk)
                # Loop over number of spikes per wb
                mi = 0
                for sp in unq
                    mask = nspike .== sp
                    nwb = sum(mask)
                    if (nwb > sp) && (estimator.k < nwb)
                        @inbounds X = StateSpaceSet(x[mask, 1:sp] .+ (noise * rand(nwb, sp)))
                        @inbounds Y = StateSpaceSet(y[mask, :])
                        mi += mutualinfo(estimator, X, Y) * probs[sp]
                    end
                end
                local_mi[i] = mi
            end
            return local_mi
        end
    end
    # Collect results from each thread
    states = fetch.(tasks) 
    # Assmeble vector of vectors into matrix
    MI = reduce(vcat, states)
    return MI
end

# Single-threaded version
function repeat_corrupted_MI_single(x::Array{Float64}, y::Array{Float64}; 
    estimator=GaoOhViswanath(k=4), 
    repeats::Int=150, 
    noise::Vector{Float64}=exp10.(range(log10(0.05), stop=log10(6), length=120)))
    # How many spikes per wingbeat from X
    nspike = vec(sum((!).(isnan.(x)), dims=2))
    # Probability of each possible number of spikes (except zero)
    unq = unique(nspike)
    unq = unq[unq .!= 0]
    probs = Dict([(u, sum(nspike .== u) / length(nspike)) for u in unq])
    # Preallocate
    MI = zeros(length(noise), repeats)
    # Loop over noise levels
    for i in eachindex(noise)
        # Loop over repeats
        for j in 1:repeats
            # Loop over number of spikes per wb
            mi = 0
            for sp in unq
                mask = nspike .== sp
                nwb = sum(mask)
                if (nwb > sp) && (estimator.k < nwb)
                    X = StateSpaceSet(x[mask, 1:sp] .+ (noise[i] * rand(nwb, sp)))
                    Y = StateSpaceSet(y[mask, :])
                    mi += mutualinfo(estimator, X, Y) * probs[sp]
                end
            end
            MI[i,j] = mi
        end
    end
    return MI
end

function subsampling_MI(x, y; 
    split_sizes=[1,2,3,4,5],
    estimator=GaoOhViswanath(k=4),
    do_plot=false)
    # Generate array of indices for each subset. Subsets have to be non-overlapping
    data_chunks = []
    for ss in split_sizes
        inds = randperm(size(x,1))
        l = round.(Int, LinRange(0, size(x,1), ss+1))
        for j in 1:ss
            push!(data_chunks, inds[l[j]+1 : l[j+1]])
        end
    end
    tasks = map(data_chunks) do chunk
        @spawn begin
            xlocal, ylocal = x[chunk,:], y[chunk,:]
            # # How many spikes per wingbeat from X
            # nspike = vec(sum((!).(isnan.(xlocal)), dims=2))
            # # Probability of each possible number of spikes (except zero)
            # unq = unique(nspike)
            # unq = unq[unq .!= 0]
            # probs = zeros(maximum(unq))
            # probs[unq] .= [sum(nspike .== u) / length(nspike) for u in unq]
            nspikeX, nspikeY = vec(sum((!).(isnan.(xlocal)), dims=2)), vec(sum((!).(isnan.(ylocal)), dims=2))
            nspike_combinations = unique(eachrow(hcat(nspikeX, nspikeY)))
            probs = Dict(combo => sum((nspikeX .== combo[1]) .&& (nspikeY .== combo[2])) / length(nspikeX) for combo in nspike_combinations)
            # Loop over number of spikes per wb
            mi = 0
            for combo in nspike_combinations
                mask = (nspikeX .== combo[1]) .&& (nspikeY .== combo[2])
                nwb = sum(mask)
                @inbounds if (nwb > sum(combo)) && (estimator.k < nwb)
                    X = StateSpaceSet(xlocal[mask, 1:combo[1]])
                    Y = StateSpaceSet(ylocal[mask, 1:combo[2]])
                    mi += mutualinfo(estimator, X, Y) * probs[combo]
                end
            end
            return mi
        end
    end
    # Collect results from each thread, rearrange
    result = fetch.(tasks) 
    subsets = inverse_rle(split_sizes, split_sizes)
    MI = [[result[i] for i in findall(subsets .== s)] for s in split_sizes]
    # Plot if asked to
    if do_plot
        f = Figure()
        ax = Axis(f[1,1], xscale=log10, xlabel="Added noise amplitude (ms)", ylabel="MI (bits)")
        μ, σ = mean.(MI), std.(MI)
        errorbars!(ax, split_sizes / size(x,1), μ, μ - σ, μ + σ, whiskerwidth=10)
        scatter!(ax, split_sizes / size(x,1), μ)
        ax.xlabel = "1/N"
        ax.ylabel = "MI (bits)"
        display(f)
    end
    return MI
end

# Somehow this spits out estimates that are just a little bit bigger than the original matlab version
# I cannot figure out why, after a few hours of testing. 
# So use with some caution, though the changes won't affect precision estimates much
# TODO: Document correctly
function find_std_subsampling(x, y; 
    split_sizes=[1,2,3,4,5],
    estimator=GaoOhViswanath(k=4),
    do_plot=false)
    MI = subsampling_MI(x, y; split_sizes, estimator=estimator)
    variances = var.(MI)
    variance_predicted = sum((split_sizes[2:end] .- 1) ./ split_sizes[2:end] .* variances[2:end]) ./ sum(split_sizes[2:end] .- 1)
    return sqrt(variance_predicted)
end

# TODO: Clean up, likely split this to have version that modifies plot and version that makes plot using multiple dispatch
function precision(x::Array{Float64}, y::Array{Float64};
    noise::Vector{Float64}=exp10.(range(log10(0.05), stop=log10(6), length=120)),
    estimator=GaoOhViswanath(k=4),
    split_sizes=collect(1:5),
    repeats=100,
    do_plot=false, 
    ax=nothing)
    # Calculate mutual information as spike times corrupted with noise
    MI = repeat_corrupted_MI(x, y; estimator=estimator, noise=noise, repeats=repeats)
    # Find std of MI at zero noise using non-overlapping subsets
    zero_noise_MI = mean(repeat_corrupted_MI(x, y; estimator=estimator, noise=[1e-6], repeats=10))
    sd = find_std_subsampling(x, y; split_sizes=split_sizes, estimator=estimator)
    meanMI = mean(MI, dims=2)
    below_threshold = meanMI .< (zero_noise_MI - sd)
    if any(below_threshold)
        ind = findfirst(below_threshold)[1]
        precision_level = noise[ind]
    else
        precision_level = NaN
    end
    # Plot if requested
    if do_plot
        if isnothing(ax)
            f = Figure()
            ax = Axis(f[1,1], xscale=log10, xlabel="Added noise amplitude (ms)", ylabel="MI (bits)")
        end
        μ, σ = vec(meanMI), vec(std(MI, dims=2))
        band!(ax, noise, μ .+ σ, μ .- σ, color=(:blue, 0.5))
        lines!(ax, noise, μ, color=:blue)
        lines!(ax, noise, fill(zero_noise_MI - sd, length(noise)), linestyle=:dash, color=:blue)
        if isnothing(ax)
            display(f)
        end
    end
    return precision_level
end
function precision_single(x::Array{Float64}, y::Array{Float64};
    noise::Vector{Float64}=exp10.(range(log10(0.05), stop=log10(6), length=120)),
    estimator=GaoOhViswanath(k=4),
    split_sizes=[1,2,3,4,5],
    repeats=100)
    # Calculate mutual information as spike times corrupted with noise
    MI = repeat_corrupted_MI_single(x, y; estimator=estimator, noise=noise, repeats=repeats)
    # Find std of MI at zero noise using non-overlapping subsets
    sd = find_std_subsampling(x, y; split_sizes=split_sizes, estimator=estimator)
    meanMI = mean(MI, dims=2)
    below_threshold = meanMI .< (meanMI[1] - sd)
    if any(below_threshold)
        ind = findfirst(below_threshold)[1]
        precision_level = noise[ind]
    else
        precision_level = NaN
    end
    return precision_level
end

"""
Version of precision function that exists primarily to draw plot
Makes new figure if ax is passed as nothing
"""
function precision_plot!(ax, x::Array{Float64}, y::Array{Float64};
    noise::Vector{Float64}=exp10.(range(log10(0.05), stop=log10(6), length=120)),
    estimator=GaoOhViswanath(k=4),
    split_sizes=collect(1:5),
    repeats=100,
    color=:blue,
    normalize_amplitude=false)
    # Calculate mutual information as spike times corrupted with noise
    MI = repeat_corrupted_MI(x, y; estimator=estimator, noise=noise, repeats=repeats)
    # Find std of MI at zero noise using non-overlapping subsets
    zero_noise_MI = mean(repeat_corrupted_MI(x, y; estimator=estimator, noise=[1e-6], repeats=10))
    sd = find_std_subsampling(x, y; split_sizes=split_sizes, estimator=estimator)
    meanMI = mean(MI, dims=2)
    if isnothing(ax)
        f = Figure()
        ax = Axis(f[1,1], xscale=log10, xlabel="Added noise amplitude (ms)", ylabel="MI (bits)")
    end
    μ, σ = vec(meanMI), vec(std(MI, dims=2))
    if normalize_amplitude
        maxval = maximum(μ)
        μ ./= maxval
        σ ./= maxval
        zero_noise_MI /= maxval
    end
    band!(ax, noise, μ .+ σ, μ .- σ, color=(color, 0.5))
    lines!(ax, noise, μ, color=color)
    lines!(ax, noise, fill(zero_noise_MI - sd, length(noise)), linestyle=:dash, color=color)
    if isnothing(ax)
        display(f)
    end
end

#--- Precision estimation methods meant to reduce MI calls
"""
fast_precision
Relies on a rough first pass, then exhaustive pass near where rough pass indicated precision point likely
"""
function fast_precision(x::Array{Float64}, y::Array{Float64};
    noise::Tuple=(0.05, 6),
    estimator=GaoOhViswanath(k=4),
    split_sizes::Array{Int}=[1,2,3,4,5],
    repeats::Int=100,
    zero_repeats::Int=5,
    rough_passes::Int=20,
    rough_repeats::Int=3,
    # fine_passes_per_OOM::Int=60,
    fine_passes::Int=20,
    fine_search_OOM=0.2)
    # Get threshold as MI at zero noise minus std. at zero noise
    zero_noise_MI = mean(repeat_corrupted_MI(x, y; estimator=estimator, noise=[1e-6], repeats=zero_repeats))
    sd = find_std_subsampling(x, y; split_sizes=split_sizes, estimator=estimator)
    threshold = zero_noise_MI - sd
    # Run rough first pass with no repeats, sparse noise
    rough_noise = exp10.(range(log10(noise[1]), stop=log10(noise[2]), length=rough_passes))
    rough_MI = repeat_corrupted_MI(x, y; estimator=estimator, noise=rough_noise, repeats=rough_repeats)
    # Find where sd crossed rough MI curve
    rough_below_threshold = mean(rough_MI, dims=2) .< threshold
    if any(rough_below_threshold)
        ind = findfirst(rough_below_threshold)[1]
        rough_precision = rough_noise[ind]
    else
        rough_precision = noise[2]
    end
    # Do thorough search centered on detected noise level
    noise_lim = log10(rough_precision) .+ [-fine_search_OOM/2, +fine_search_OOM/2] # On log scale!
    noise = exp10.(range(noise_lim[1], stop=noise_lim[2], length=fine_passes))
    MI = repeat_corrupted_MI(x, y; estimator=estimator, noise=noise, repeats=repeats)
    # Find precision level
    meanMI = mean(MI, dims=2)
    below_threshold = meanMI .< threshold
    if any(below_threshold)
        ind = findfirst(below_threshold)[1]
        precision_level = noise[ind]
    else
        precision_level = NaN
    end
    return precision_level
end
"""
fast_precision
Relies on a rough first pass, then a modified bisection method (like root finding) to narrow to 
likely precision point within a given tolerance
"""
# TODO: Have special return for when tol not reached
# TODO: Prettier code for recursive call back to rough pass
# TODO: Version with secant, false position, or brent's method
function fast_precision_bisect(x::Array{Float64}, y::Array{Float64};
    noise::Tuple=(0.05, 6),
    estimator=GaoOhViswanath(k=4),
    split_sizes::Array{Int}=collect(1:8),
    zero_repeats::Int=5,
    rough_passes::Int=30, rough_repeats::Int=3,
    interp_times=4,
    fine_repeats::Int=300, fine_search_OOM=0.5,
    tol=1e-3,
    max_iter=20,
    do_plot=false,
    print_evals=false)
    # Get threshold as MI at zero noise minus std. at zero noise. 
    # Repeated and mean taken to get stable value for sometimes unstable situations
    zero_noise_MI = mean(repeat_corrupted_MI(x, y; estimator=estimator, noise=[1e-6], repeats=zero_repeats))
    sd = find_std_subsampling(x, y; split_sizes=split_sizes, estimator=estimator)
    threshold = zero_noise_MI - sd
    # Run rough first pass with no repeats, sparse noise
    rough_noise = exp10.(range(log10(noise[1]), stop=log10(noise[2]), length=rough_passes))
    rough_MI = repeat_corrupted_MI(x, y; estimator=estimator, noise=rough_noise, repeats=rough_repeats)
    if rough_repeats > 1
        rough_MI = mean(rough_MI, dims=2)
    end
    rough_MI = vec(rough_MI)
    sg_window = 2 * floor(Int, rough_passes / 4) + 1 # Rounds rough_passes/2 to nearest odd int
    rough_MI_smooth = savitzky_golay(rough_MI, sg_window, 2).y
    # Interpolate to (rough_passes * interp_times) more points to improve first guess
    new_rough_noise = LinRange(rough_noise[1], rough_noise[end], round(Int, rough_passes * interp_times))
    rough_MI_smooth = linterp(rough_noise, rough_MI_smooth, new_rough_noise)
    # Find where sd crossed rough MI curve
    rough_below_threshold = rough_MI_smooth .< threshold
    if any(rough_below_threshold)
        rough_precision = new_rough_noise[findfirst(rough_below_threshold)[1]]
        init_calls = rough_repeats * rough_passes + zero_repeats + sum(split_sizes)
    else
        # If no crossing, perform another rough pass at higher noise 
        # Could do this recursive, but I'd rather have it just terminate at one more pass
        noise = (noise[2], noise[2] + exp10(log10(noise[2]) - log10(noise[1])))
        rough_noise = exp10.(range(log10(noise[1]), stop=log10(noise[2]), length=rough_passes))
        rough_MI = repeat_corrupted_MI(x, y; estimator=estimator, noise=rough_noise, repeats=rough_repeats)
        if rough_repeats > 1
            rough_MI = mean(rough_MI, dims=2)
        end
        rough_MI = vec(rough_MI)
        rough_MI_smooth = savitzky_golay(rough_MI, sg_window, 2).y
        # Interpolate to (rough_passes * interp_times) more points to improve first guess
        new_rough_noise = LinRange(rough_noise[1], rough_noise[end], round(Int, rough_passes * interp_times))
        rough_MI_smooth = linterp(rough_noise, rough_MI_smooth, new_rough_noise)
        # Find where sd crossed rough MI curve
        rough_below_threshold = rough_MI_smooth .< threshold
        if any(rough_below_threshold)
            rough_precision = new_rough_noise[findfirst(rough_below_threshold)[1]]
        else
            # Simply return NaN if still not found
            return NaN
        end
        init_calls = 2 * rough_repeats * rough_passes + zero_repeats + sum(split_sizes)
    end
    # Plot of initial heuristic for illustration purposes
    if do_plot
        f = Figure()
        ax = Axis(f[1,1], xscale=log10)
        lines!(ax, rough_noise, vec(rough_MI))
        lines!(ax, new_rough_noise, rough_MI_smooth)
        lines!(ax, [rough_noise[1], rough_noise[end]], [threshold, threshold])
        display(f)
    end
    # Do bisection search 
    bounds = exp10.([log10(rough_precision) - fine_search_OOM/2, log10(rough_precision) + fine_search_OOM/2])
    # test_noise = mean(bounds)
    test_noise = rough_precision
    i = 1
    while i <= max_iter
        # Evaluate at half of bounds
        local mi = repeat_corrupted_MI(x, y, test_noise; estimator=estimator, repeats=fine_repeats)
        res = mean(mi)
        # Exit if within tolerance
        if abs(res - threshold) <= tol
            break
        end
        # Otherwise assign noise we just checked as a new bound, set up new noise level to test, repeat
        change_index = res >= threshold ? 1 : 2
        bounds[change_index] = test_noise
        test_noise = mean(bounds)
        i += 1
    end
    if print_evals
        bisect_calls = fine_repeats * (i - 1)
        println("$init_calls initial MI calls and $bisect_calls during search")
    end
    return test_noise
end

function fast_precision_bisect_only(x::Array{Float64}, y::Array{Float64};
    noise::Tuple=(0.05, 6),
    estimator=GaoOhViswanath(k=4),
    split_sizes::Array{Int}=[1,2,3,4,5],
    repeats::Int=150,
    tol=1e-4,
    max_iter=40)
    # Get threshold as MI at zero noise minus std. at zero noise
    zero_noise_MI = spike_MI(x, y; estimator=estimator)
    sd = find_std_subsampling(x, y; split_sizes=split_sizes, estimator=estimator)
    threshold = zero_noise_MI - sd
    # Do bisection search 
    bounds = [noise[1], noise[2]]
    test_noise = mean(bounds)
    i = 1
    while i <= max_iter
        # Evaluate at half of bounds
        local mi = repeat_corrupted_MI(x, y, test_noise; estimator=estimator, repeats=repeats)
        res = mean(mi)
        # Exit if within tolerance
        if abs(res - threshold) <= tol
            break
        end
        # Otherwise assign noise we just checked as a new bound, set up new noise level to test, repeat
        change_index = res >= threshold ? 1 : 2
        bounds[change_index] = test_noise
        test_noise = mean(bounds)
        i += 1
    end
    return test_noise
end



"""
Caller for precision when data is in typical dataframe format (single column arrays, long format)

Assumes you pass in columns corresponding to one single muscle, and that wingbeats are unique
Assumes a count column exists
Assumes last however many columns are PCs of force and/or torque
"""
function precision_dataframe(wb, count, time, args...;
    estimator=GaoOhViswanath(k=4), 
    noise::Tuple=(0.05, 6),
    split_sizes=collect(1:8),
    rough_passes::Int=200,
    fine_repeats::Int=200,
    fine_search_OOM=0.1,
    tol=1e-3,
    max_iter=20,
    max_wb=nothing)
    # Get how many wingbeats, max number of spikes, how many PCs
    uniquewb = unique(wb)
    nwb = length(uniquewb) 
    nspike = maximum(count)
    npc = length(args)
    # Preallocate new arrays to rearrange values into
    x = fill(NaN, nwb, nspike)
    y = fill(0.0, nwb, npc)
    # Get indices of each wingbeat, start filling arrays
    indices_dict = group_indices(wb)
    for (i, (value, indices)) in enumerate(indices_dict)
        x[i,1:length(indices)] = time[indices]
        y[i,:] = [first(col[indices]) for col in args]
    end
    if !isnothing(max_wb) && max_wb < nwb
        inds = rand(1:nwb, max_wb)
        x = x[inds,:]
        y = y[inds,:]
    end
    prec = fast_precision_bisect(x, y; 
        estimator=estimator, noise=noise, split_sizes=split_sizes,
        rough_passes=rough_passes,
        fine_repeats=fine_repeats,
        fine_search_OOM=fine_search_OOM,
        tol=tol,
        max_iter=max_iter)
    return prec
end
# Just the rearrange to x and y part
function dataframe_to_XYarray(wb, count, time, args...; max_wb=nothing)
    # Get how many wingbeats, max number of spikes, how many PCs
    uniquewb = unique(wb)
    nwb = length(uniquewb) 
    nspike = maximum(count)
    npc = length(args)
    # Preallocate new arrays to rearrange values into
    x = fill(NaN, nwb, nspike)
    y = fill(0.0, nwb, npc)
    # Get indices of each wingbeat, start filling arrays
    indices_dict = group_indices(wb)
    for (i, (value, indices)) in enumerate(indices_dict)
        x[i,1:length(indices)] = time[indices]
        y[i,:] = [first(col[indices]) for col in args]
    end
    if !isnothing(max_wb) && max_wb < nwb
        inds = rand(1:nwb, max_wb)
        x = x[inds,:]
        y = y[inds,:]
    end
    return x, y
end

function XY_array_from_dataframe(targetcols, unit, values, wb)
    mask1, mask2 = unit .== targetcols[1], unit .== targetcols[2]
    # Get wingbeats that have both units/targets spiking
    wb1, wb2 = wb[mask1], wb[mask2]
    common_wingbeats = find_common_elements(wb1, wb2)
    dim1, dim2 = max_count(wb1), max_count(wb2)
    # Could be more efficient here but whatever
    wbinds1, wbinds2 = group_indices(wb1), group_indices(wb2)
    # Preallocate 
    X = fill(NaN, length(common_wingbeats), dim1)
    Y = fill(NaN, length(common_wingbeats), dim2)
    # Loop over wingbeats
    for (i, iwb) in enumerate(common_wingbeats)
        data1, data2 = values[mask1][wbinds1[iwb]], values[mask2][wbinds2[iwb]]
        X[i,1:length(data1)] = data1
        Y[i,1:length(data2)] = data2
    end
    return X, Y
end


"""
linterp, linear interpolation
Utility function, really basic linear interpolation. 
Assumes new points are within x range of original points,
Assumes new points are monotonically increasing
"""
function linterp(x::AbstractVector{T}, y::AbstractVector{T}, x_new::AbstractVector{T}) where T
    @inbounds begin
        y_new = zeros(length(x_new))
        idx0, idx1 = 1, 2 
        for i in eachindex(y_new)
            if x_new[i] > x[idx1]
                idx0 = searchsortedlast(x, x_new[i])
                idx1 = idx0 + 1
            end
            y_new[i] = y[idx0] + (y[idx1] - y[idx0]) / (x[idx1] - x[idx0]) * (x_new[i] - x[idx0])
        end
        return y_new
    end
end