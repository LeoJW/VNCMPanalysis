using HDF5
using Statistics
using GLMakie
using AlgebraOfGraphics
using DataFrames
using DataFramesMeta
using Pipe
using SavitzkyGolay

function find_precision(noise_levels, mi, lower_bound=5e-4)
    curve = vec(mean(mi, dims=1))
    deriv = savitzky_golay(curve, 15, 2; deriv=2).y
    bound_mask = noise_levels .>= lower_bound
    return noise_levels[bound_mask][findmin(deriv[bound_mask])[2]]
end


analysis_dir = @__DIR__
data_dir = joinpath(analysis_dir, "..", "localdata", "estimation_runs")

file = "network_arch_comparison_PACE_2025-06-05.h5"

function read_network_arch_file!(df, file, task)
    precision_noise = h5read(joinpath(data_dir, file), "dict_0")
    precision_curves = h5read(joinpath(data_dir, file), "dict_1")
    time_per_epoch = h5read(joinpath(data_dir, file), "dict_2")
    # Construct dataframe
    first_row = split(first(keys(time_per_epoch)), "_")
    names = vcat(first_row[1:2:end], ["time", "mi", "precision", "precision_curve", "precision_noise"])
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat([x ? Float64 : String for x in is_numeric], Float64, Float64, Float64, Matrix{Float64}, Vector{Float64})
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(time_per_epoch)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        vals[6] = task
        push!(thisdf, vcat(
            vals, 
            time_per_epoch[key], 
            mean(precision_curves[key], dims=1)[1], 
            find_precision(precision_noise[key] .* 0.0001 .* 1000, precision_curves[key]),
            [precision_curves[key] .* log2(exp(1)) ./ (512 * 0.0001)],
            [precision_noise[key] .* 0.0001 .* 1000]
        ))
    end
    append!(df, thisdf)
end


df = DataFrame()
for task in 0:5
    read_network_arch_file!(df, joinpath(data_dir, "network_arch_comparison_PACE_task_$(task)_2025-06-05.h5"), task)
end

# Remove obvious failures, convert units
df = @pipe df |> 
@subset(_, :mi .> 0.05) |> 
@transform(_, :mi = :mi .* log2(exp(1)) ./ (512 * 0.0001))

##

@pipe df |> 
@subset(_, :neuron .== "all") |> 
@transform(_, :layout = ifelse.(:layout .== "None", "0", :layout)) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:time, :mi, col=:layout, color=:layers, row=:filters=>nonnumeric) * 
visual(Scatter)
) |> 
draw(_)

##

@pipe df |> 
@subset(_, :neuron .== "all") |> 
@transform(_, :layout = ifelse.(:layout .== "None", "0", :layout)) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:precision, :mi, col=:layout, color=:layers, row=:filters=>nonnumeric) * 
visual(Scatter)
) |> 
draw(_)

##
dt = @pipe df |> 
@subset(_, (:layout .!= "0") .&& (:layout .!= "all")) |> 
@subset(_, :layers .>= 5) |> 
@subset(_, :neuron .== "all")

f = Figure()
for (i,gdf) in enumerate(groupby(dt, :layout))
    for (j,ggdf) in enumerate(groupby(gdf, :filters))
        ax = Axis(f[j,i], xscale=log10, title=first(ggdf.layout) * " " * string(ggdf.filters[1]))
        for row in eachrow(ggdf)
            curve = vec(mean(row.precision_curve, dims=1))
            deriv = savitzky_golay(curve, 15, 2; deriv=2).y
            lines!(ax, row.precision_noise, curve, color=row.layers, colorrange=(5,7))
        end
        # ylims!(ax, 0, maximum(dt.mi))
        xlims!(ax, 10^-1, 10^2)
    end
end
f

## Theoretical max receptive field calculations
RFs, RFdlin, RFdmult, RFdexp = 1, 1, 1, 1
k = 3
layers = 7
s = vcat(1, fill(2, layers-1))
sc = cumsum(s)
dlin = collect(1:layers) # Linear dilation
dmult = 2 .* collect(1:layers) # Multiplying dilation
dexp = 2 .^ collect(0:layers-1) # Exponential dilation
for i in 1:layers
    println("------ Layer $(i) -------")
    RFs = RFs + (k - 1) * sc[i]
    println("Standard : $(RFs)")
    RFdlin = RFdlin + (k - 1) * dlin[i] * sc[i]
    RFdmult = RFdmult + (k - 1) * dmult[i] * sc[i]
    RFdexp = RFdexp + (k - 1) * dexp[i] * sc[i]
    println("Linear Dilation : $(RFdlin)")
    println("Multiplicative Dilation : $(RFdmult)")
    println("Exponential Dilation : $(RFdexp)")
end

##

period = 0.0001
x = vcat(0, logrange(period, 0.04, 20) .* 1000)
curve = vec(mean(precision_curves[first(keys(precision_curves))], dims=1))
deriv = savitzky_golay(curve, 5, 2; deriv=2).y

f = Figure()
ax = Axis(f[1,1], xscale=log10)
lines!(ax, x, deriv)
lines!(ax, x, curve)
min_idx = findmin(deriv)[2]
scatter!(ax, x[min_idx], deriv[min_idx])
scatter!(ax, x[min_idx], curve[min_idx])
f



## --- Subsampling results

file = "subsampling_PACE_2025-06-04.h5"
subsamples = h5read(joinpath(data_dir, file), "dict_0")
mi = h5read(joinpath(data_dir, file), "dict_1")

moths = keys(mi)

f = Figure()
ax = [Axis(f[i,1], title=moth) for (i,moth) in enumerate(moths)]
for (i,moth) in enumerate(moths)
    scatter!(ax[i], subsamples[moth], mi[moth])
end
linkyaxes!(ax)
f


## --- Binning results

file = "binning_PACE_2025-06-05.h5"

precision_noise = h5read(joinpath(data_dir, file), "dict_0")
precision_curves = h5read(joinpath(data_dir, file), "dict_1")

# Construct dataframe
first_row = split(first(keys(precision_noise)), "_")
names = vcat(first_row[1:2:end], ["mi", "precision"])
is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]], true, true)
types = [x ? Float64 : String for x in is_numeric]

df = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
df = df[!, Symbol.(names)] # Undo name sorting
for key in keys(precision_noise)
    keysplit = split(key, "_")[2:2:end]
    vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
    if all(mean(precision_curves[key], dims=1)[2:end] .< 0)
        prec = NaN
    else
        prec = find_precision(precision_noise[key] .* vals[end-1], precision_curves[key])
    end
    push!(df, vcat(vals, mean(precision_curves[key], dims=1)[1], prec))
end


##

@pipe df |> 
@transform(_, :mi = :mi ./ :window * 1000 .* log2(exp(1))) |> 
@transform(_, :period = :period .* 1000) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:period, :mi=>"I(X,Y) (bits/s)", color=:window=>nonnumeric, col=:neuron) * visual(Scatter)
) |> 
draw(_, facet=(; linkyaxes=:none), axis=(; xscale=log10))

##

@pipe df |> 
@transform(_, :mi = :mi ./ :window * 1000 .* log2(exp(1))) |> 
@transform(_, :samps_per_window = :window ./ (:period .* 1000)) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:samps_per_window, :mi=>"I(X,Y) (bits/s)", color=:window=>nonnumeric, col=:neuron) * visual(Scatter)
) |> 
draw(_, facet=(; linkyaxes=:none), axis=(; xscale=log10))

##

@pipe df |> 
@transform(_, :period = :period .* 1000) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:period, :precision, color=:window=>nonnumeric, col=:neuron) * visual(Scatter)
) |> 
draw(_, 
    # facet=(; linkyaxes=:none), 
    axis=(; xscale=log10))

##

min = findmin(df.period)
idx = findall(df.period .== min[1])[4]
f = Figure()
ax = Axis(f[1,1], xscale=log10)
key = collect(keys(precision_curves))[idx]
lines!(ax, vec(precision_noise[key])[2:end] .* df.period[idx], vec(mean(precision_curves[key], dims=1))[2:end])
f

