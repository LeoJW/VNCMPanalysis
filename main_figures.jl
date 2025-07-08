using HDF5
using CSV
using Statistics
using GLM
using CairoMakie
using GLMakie
using AlgebraOfGraphics
using DataFrames
using DataFramesMeta
using Pipe


analysis_dir = @__DIR__
data_dir = joinpath(analysis_dir, "..", "localdata", "estimation_runs")
dark_or_light = "light"
if dark_or_light .== "light"
    set_theme!()
    fig_dir = joinpath(analysis_dir, "figs_light")
else
    set_theme!(theme_dark())
    fig_dir = joinpath(analysis_dir, "figs_dark")
end

function find_precision_deriv(noise_levels, mi; lower_bound=5e-4)
    sg_window = 2 * floor(Int, length(noise_levels) / 5) + 1
    sg_window = sg_window < 2 ? 5 : sg_window
    curve = vec(mean(mi, dims=1))
    deriv = savitzky_golay(curve, sg_window, 2; deriv=2).y
    bound_mask = deriv .<= (- lower_bound)
    if !isempty(deriv[bound_mask])
        return noise_levels[bound_mask][findmin(deriv[bound_mask])[2]]
    else
        return NaN
    end
end

function find_precision_threshold(noise_levels, mi; threshold=0.9)
    curve = mi ./ mi[1]
    ind = findfirst(curve .< threshold)
    if isnothing(ind)
        return NaN
    else
        return noise_levels[ind]
    end
end


function read_network_arch_file!(df, file, task)
    precision_noise = h5read(joinpath(data_dir, file), "dict_0")
    precision_curves = h5read(joinpath(data_dir, file), "dict_1")
    time_per_epoch = h5read(joinpath(data_dir, file), "dict_2")
    # Construct dataframe
    first_row = split(first(keys(time_per_epoch)), "_")
    names = vcat(first_row[1:2:end], ["time", "mi", "precision", "precision_curve", "precision_noise"])
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat([x ? Float64 : String for x in is_numeric], Float64, Float64, Float64, Vector{Float64}, Vector{Float64})
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(time_per_epoch)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        vals[9] = task
        push!(thisdf, vcat(
            vals,
            time_per_epoch[key], 
            precision_curves[key][1] .* log2(exp(1)), 
            find_precision_threshold(precision_noise[key] .* 1000, precision_curves[key][2:end]),
            [precision_curves[key] .* log2(exp(1))],
            [precision_noise[key] .* 1000]
        ))
    end
    append!(df, thisdf)
end