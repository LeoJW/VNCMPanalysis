using HDF5
using Statistics
using GLMakie
using AlgebraOfGraphics
using DataFrames
using DataFramesMeta
using Pipe
using SavitzkyGolay

function find_precision(noise_levels, mi; lower_bound=5e-4)
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

analysis_dir = @__DIR__
data_dir = joinpath(analysis_dir, "..", "localdata", "estimation_runs")

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
            find_precision(precision_noise[key] .* 0.0001 .* 1000, precision_curves[key], lower_bound=0),
            [precision_curves[key] .* log2(exp(1)) ./ (512 * 0.0001)],
            [precision_noise[key] .* 0.0001 .* 1000]
        ))
    end
    append!(df, thisdf)
end


df = DataFrame()
for task in 0:5
    read_network_arch_file!(df, joinpath(data_dir, "network_arch_comparison_PACE_task_$(task)_2025-06-08.h5"), task)
end

# Remove obvious failures, convert units
df = @pipe df |> 
# @subset(_, :mi .> 0.05) |> 
@transform(_, :mi = :mi .* log2(exp(1)) ./ (512 * 0.0001))

##

@pipe df |> 
@subset(_, :neuron .== "all") |> 
@transform(_, :layout = ifelse.(:layout .== "None", "0", :layout)) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:time, :precision, col=:layout, color=:layers, row=:filters=>nonnumeric) * 
visual(Scatter)
) |> 
draw(_)#, axis=(; limits=((2, 20), nothing)))

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
@subset(_, :neuron .== "neuron")

f = Figure()
for (i,gdf) in enumerate(groupby(dt, :layout))
    for (j,ggdf) in enumerate(groupby(gdf, :filters))
        ax = Axis(f[j,i], xscale=log10, title=first(ggdf.layout) * " " * string(ggdf.filters[1]))
        for row in eachrow(ggdf)
            curve = vec(mean(row.precision_curve, dims=1))
            sg_window = 2 * floor(Int, length(row.precision_noise) / 5) + 1
            sg_window = sg_window < 2 ? 5 : sg_window
            deriv = savitzky_golay(curve, sg_window, 2; deriv=2).y
            bound_mask = deriv .<= (- 3e-4)
            lines!(ax, row.precision_noise[bound_mask], deriv[bound_mask], color=row.layers, colorrange=(5,7))
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


## ------------------------------------ Subsampling results ------------------------------------

file = "subsampling_PACE_2025-06-06.h5"
subsamples = h5read(joinpath(data_dir, file), "dict_0")
mi = h5read(joinpath(data_dir, file), "dict_1")
dim = h5read(joinpath(data_dir, file), "dict_2")

moths = collect(keys(mi))
unique_dims = unique(dim[moths[1]])
# Assemble into dataframe
df = DataFrame()
for moth in moths
    ss = repeat(subsamples[moth], inner=length(unique_dims))
    append!(df, DataFrame(subsample=ss, mi=mi[moth], dim=dim[moth], moth=moth))
end


f = Figure()
ax = [Axis(f[i,1], xticks=0:9) for (i,moth) in enumerate(moths)]
for (i,groupdf) in enumerate(groupby(df, :moth))
    thisdf = @pipe groupdf |> 
        @transform(_, :mi = :mi ./ (512 .* 0.0001)) |> 
        groupby(_, [:subsample, :dim]) |> 
        combine(_, :mi => mean => :mi, :mi => std => :mi_std)
    
    scatter!(ax[i], thisdf.subsample, thisdf.mi, color=thisdf.dim, colorrange=extrema(unique_dims))
    errorbars!(ax[i], thisdf.subsample, thisdf.mi, thisdf.mi_std, color=thisdf.dim, colorrange=extrema(unique_dims))
    for dt in groupby(thisdf, :dim)
        lines!(ax[i], dt.subsample, dt.mi, color=dt.dim, colorrange=extrema(unique_dims))
    end
    text!(ax[i], 0.5, 1, text=groupdf.moth[1], font=:bold, align=(:center, :top), space=:relative)
    xlims!(ax[i], 0, nothing)
    ylims!(ax[i], 0, nothing)
end
ax[end].xlabel =  "No. of Subsamples"
linkyaxes!(ax)
[hidexdecorations!(a, grid=false) for a in ax[1:end-1]]
rowgap!(f.layout, 10)
Label(f[:, 0], "I(X,Y) (bits/s)", rotation = pi/2)
Colorbar(f[:,2], colorrange=extrema(unique_dims), ticks=unique_dims, label="Dimensionality")
f

##

thisdf = @pipe df |> 
    @subset(_, :subsample .== 1) |> 
    @transform(_, :mi = :mi ./ (512 .* 0.0001)) |> 
    groupby(_, [:subsample, :dim, :moth]) |> 
    combine(_, :mi => mean => :mi)
f = Figure()
ax = [Axis(f[i,1]) for i in eachindex(moths)]
for (i, gdf) in enumerate(groupby(thisdf, :moth))
    println(gdf.mi)
    # errorbars!(ax[i,1], gdf.dim, gdf.mi, gdf.mi_std)
    lines!(ax[i,1], gdf.dim, gdf.mi)
    scatter!(ax[i,1], gdf.dim, gdf.mi)
    ax[i].xticks = unique(gdf.dim)
    text!(ax[i], 0.5, 1, text=gdf.moth[1], font=:bold, align=(:center, :top), space=:relative)
end
ax[end].xlabel =  "Dimensionality"
linkyaxes!(ax)
[hidexdecorations!(a, grid=false) for a in ax[1:end-1]]
rowgap!(f.layout, 10)
Label(f[:, 0], "I(X,Y) (bits/s)", rotation = pi/2)
f

## ------------------------------------ Binning results ------------------------------------

function read_binning_file!(df, file, task)
    precision_noise = h5read(file, "dict_0")
    precision_curves = h5read(file, "dict_1")
    # Construct dataframe
    first_row = split(first(keys(precision_noise)), "_")
    names = vcat(first_row[1:2:end], ["mi", "precision", "precision_curve", "precision_noise"])
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat([x ? Float64 : String for x in is_numeric], Float64, Float64, Matrix{Float64}, Vector{Float64})
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(precision_noise)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        window, period = vals[2], vals[3]
        vals[4] = task + 1
        mean_curve = mean(precision_curves[key], dims=1)
        if mean_curve[end] > mean_curve[1]
            continue
        end
        if (all(mean_curve[2:end] .< 0)) || (length(precision_noise[key]) < 10)
            prec = NaN
        else
            prec = find_precision(precision_noise[key] .* period .* 1000, precision_curves[key], lower_bound=0)
        end
        push!(thisdf, vcat(
            vals,
            mean_curve[1], 
            prec,
            [precision_curves[key] .* log2(exp(1)) ./ window .* 1000],
            [precision_noise[key] .* period .* 1000]
        ))
    end
    append!(df, thisdf)
end


df = DataFrame()
for task in 0:9
    # read_binning_file!(df, joinpath(data_dir, "binning_PACE_task_$(task)_2025-06-10.h5"), task)
    read_binning_file!(df, joinpath(data_dir, "2025-06-12_binning_PACE_task_$(task).h5"), task)
end
# Convert units
df = @pipe df |> 
@transform(_, :mi = :mi ./ :window .* 1000 .* log2(exp(1))) |> 
@transform(_, :period = :period .* 1000) |> 
@transform(_, :samps_per_window = round.(Int, :window ./ :period))

##

row = rand(eachrow(df))
f = Figure()
ax = Axis(f[1,1], xscale=log10)
lines!(ax, row.precision_noise, vec(mean(row.precision_curve, dims=1)))
f

## Means by period

group_bins = sort(unique(df.period)) .- 0.001
tempdf = @pipe df |> 
@transform(_, :mi = :mi .* (:window ./ 1000)) |> 
groupby(_, [:window, :neuron, :period]) |> 
combine(_, :mi => mean => :mi, :mi => std => :mi_std, :precision => (x->mean(x[(!).(isnan.(x))])) => :precision) |> 
@transform(_, :newperiod = searchsortedlast.(Ref(group_bins), :period)) |> 
@transform(_, :logperiod = log.(:period))

@pipe tempdf[sortperm(tempdf.window),:] |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window, :mi, color=:logperiod, col=:neuron, group=:newperiod=>nonnumeric) * (visual(Scatter) + visual(Lines))
# mapping(:window, :mi, color=:logperiod, col=:neuron, group=:newperiod=>nonnumeric) * visual(Scatter)
) |> 
draw(_, facet=(; linkxaxes=:none, linkyaxes=:none))#, axis=(; yscale=log10))


## Precision vs period

group_bins = sort(unique(df.period)) .- 0.001
tempdf = @pipe df |> 
    @transform(_, :mi = :mi .* (:window ./ 1000)) |> 
    groupby(_, [:window, :neuron, :period]) |> 
    combine(_, :mi => mean => :mi, :mi => std => :mi_std, :precision => (x->mean(x[(!).(isnan.(x))])) => :precision) |> 
    @transform(_, :logperiod = log.(:period)) |> 
    @transform(_, :order = searchsortedlast.(Ref(group_bins), :period))
@pipe tempdf[sortperm(tempdf.order),:] |> 
(
AlgebraOfGraphics.data(_) *
mapping(:period, :mi, color=:window, col=:neuron, group=:window=>nonnumeric) * (visual(Scatter) + visual(Lines))
) |> 
draw(_, facet=(; linkxaxes=:none, linkyaxes=:none), axis=(; xscale=log10))

##

@pipe df |> 
@transform(_, :mi = :mi .* (:window ./ 1000)) |> 
# groupby(_, [:window, :samps_per_window, :neuron, :period]) |> 
# combine(_, :mi => mean => :mi, :precision => mean => :precision) |> 
@transform(_, :logperiod = log.(:period)) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window, :mi, color=:logperiod, col=:neuron) * visual(Scatter)
# mapping(:samps_per_window, :mi, color=:logperiod, row=:window=>nonnumeric, col=:neuron) * visual(Scatter)
# mapping(:period, :precision, color=:window, col=:neuron) * visual(Scatter)
) |> 
draw(_, facet=(; linkyaxes=:none), axis=(; xscale=log10, limits=(nothing, (0,nothing))))

## samples per window vs precision/mi
@pipe df |> 
# @subset(_, :neuron .== "all") |> 
@subset(_, :mi .> 0) |> 
@subset(_, (!).(isnan.(:precision))) |> 
groupby(_, [:window, :samps_per_window, :neuron, :period]) |> 
combine(_, :mi => mean => :mi, :precision => mean => :precision) |> 
@transform(_, :mi = :mi .* (:window ./ 1000)) |> 
@transform(_, :logperiod = log.(:period)) |> 
# @transform(_, :precision = :precision ./ (:period ./ 1000) ./ 1000) |> 
@transform(_, :samps_per_window = log10.(:samps_per_window)) |> 
# stack(_, [:precision, :mi]) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:samps_per_window=>"log10(Samples per window)", :mi=>"I(X,Y) (bits/window)", color=:window=>nonnumeric, col=:neuron, group=:window=>nonnumeric) * (visual(Scatter) + linear())
) |> 
draw(_, facet=(; linkyaxes=:none))#, axis=(; xscale=log10))


##

# row = rand(eachrow(@subset(df, (:precision .> 10))))
row = rand(eachrow(@subset(df, (:neuron .== "all") .&& (:window .== 25) .&& (:precision .< 10))))

f = Figure()
ax = Axis(f[1,1], xscale=log10, title="$(row.neuron) window $(row.window) period $(row.period)")
lines!(ax, row.precision_noise, vec(mean(row.precision_curve, dims=1)))
scatter!(ax, row.precision_noise, vec(mean(row.precision_curve, dims=1)))
f


##

min = findmin(df.period)
idx = findall(df.period .== min[1])[4]
f = Figure()
ax = Axis(f[1,1], xscale=log10)
key = collect(keys(precision_curves))[idx]
lines!(ax, vec(precision_noise[key])[2:end] .* df.period[idx], vec(mean(precision_curves[key], dims=1))[2:end])
f



## ------------------------------------ Precision fixed results ------------------------------------

# file = joinpath(data_dir, "precision_groundtruth_PACE_2025-06-09.h5")
function find_precision(noise_levels, mi; lower_bound=5e-4)
    sg_window = 2 * floor(Int, length(noise_levels) / 5) + 1
    sg_window = sg_window < 2 ? 5 : sg_window
    curve = vec(mean(mi, dims=1))
    deriv = savitzky_golay(curve, sg_window, 2; deriv=2).y
    bound_mask = deriv .<= - lower_bound
    if !isempty(deriv[bound_mask])
        return noise_levels[bound_mask][findmin(deriv[bound_mask])[2]]
    else
        return NaN
    end
end

function read_precision_file(file)
    # Returns in units of bits/window, time units in ms
    precision_noise = h5read(file, "dict_0")
    precision_curves = h5read(file, "dict_1")
    precision_noise_y = h5read(file, "dict_2")
    precision_curves_y = h5read(file, "dict_3")
    
    period = 0.0001
    
    # Construct dataframe
    added_names = [
        "mi", "precision_est", "mi_y", "precision_est_y", 
        "precision_curve", "precision_noise", 
        "precision_curve_y", "precision_noise_y"]
    added_types = [
        Float64, Float64, Float64, Float64, 
        Matrix{Float64}, Vector{Float64}, 
        Matrix{Float64}, Vector{Float64}]

    first_row = split(first(keys(precision_noise)), "_")
    last_ind = (length(first_row) ÷ 2) * 2

    names = vcat(first_row[1:2:last_ind], added_names)
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat([x ? Float64 : String for x in is_numeric], added_types)
    print(names)
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(precision_noise)
        keysplit = split(key, "_")[2:2:last_ind]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        mean_curve = mean(precision_curves[key], dims=1) .* log2(exp(1))
        mean_curve_y = mean(precision_curves_y[key], dims=1) .* log2(exp(1))
        prec = ifelse(
            all(mean_curve[2:end] .< 0), 
            NaN,
            find_precision(precision_noise[key] .* period .* 1000, precision_curves[key], lower_bound=5e-5))
        prec_y = ifelse(
            all(mean_curve_y[2:end] .< 0),
            NaN,
            find_precision(precision_noise_y[key] .* period .* 1000, precision_curves_y[key], lower_bound=5e-5))
        push!(thisdf, vcat(
            vals,
            mean_curve[1], prec,
            mean_curve_y[1], prec_y,
            [precision_curves[key] .* log2(exp(1))], [precision_noise[key] .* period .* 1000],
            [precision_curves_y[key] .* log2(exp(1))], [precision_noise_y[key] .* period .* 1000]
        ))
    end
    return thisdf
end


df = read_precision_file(joinpath(data_dir, "precision_groundtruth_PACE_2025-06-09.h5"))

##

@pipe df |> 
@transform(_, :precision = :precision .* 1000) |> 
# @subset(_, :setOn .== "muscles") |> 
@subset(_, :neuron .== "neuron") |> 
(
AlgebraOfGraphics.data(_) * 
(mapping(:precision, :mi, row=:moth, col=:setOn, color=:setOn) * 
visual(Scatter)) +
mapping([0], [1]) * visual(ABLines)
) |> 
draw(_)

##

dt = @subset(df, (:moth .== "2025-03-11") .&& (:setOn .== "neurons"))

f = Figure()
ax = Axis(f[1,1], xscale=log10)

for row in eachrow(dt)
    yvals = vec(mean(row.precision_curve[:,2:end], dims=1))

    sg_window = 2 * floor(Int, length(row.precision_noise[2:end]) / 5) + 1
    sg_window = sg_window < 2 ? 5 : sg_window
    # yvals = savitzky_golay(yvals, sg_window, 2; deriv=2).y

    bound_mask = yvals .<= -5e-5

    lines!(ax, 
        row.precision_noise[2:end], yvals, 
        # row.precision_noise[2:end][bound_mask], yvals[bound_mask], 
        color=row.precision, colorrange=extrema(dt.precision))
    # if ~isempty(yvals[bound_mask])
    #     idx = findmin(yvals[bound_mask])[2]
    #     scatter!(ax, row.precision_noise[2:end][bound_mask][idx], yvals[bound_mask][idx])
    # end
end
# xlims!(ax, 10^1, 10^2)
f