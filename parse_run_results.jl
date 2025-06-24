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
using SavitzkyGolay

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

function find_precision_threshold(noise_levels, mi)
    curve = vec(mean(mi, dims=1))
    threshold = mean(mi[:,2]) - (4 .* std(mi[:,2]))
    cross = curve .< threshold
    if !any(cross)
        return NaN
    end
    return noise_levels[findfirst(cross)]
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
            1,
            # find_precision_threshold(precision_noise[key] .* 1000, precision_curves[key] .* log2(exp(1))),
            [precision_curves[key] .* log2(exp(1))],
            [precision_noise[key] .* 1000]
        ))
    end
    append!(df, thisdf)
end


df = DataFrame()
for task in 0:5
    read_network_arch_file!(df, joinpath(data_dir, "2025-06-22_network_comparison_PACE_task_$(task).h5"), task)
end

# Remove obvious failures, convert units
# df = @pipe df |> 
# @subset(_, :mi .> 0.0)
# @transform(_, :mi = :mi .* log2(exp(1)))

# Save to CSV for R and other things
@pipe df |> 
    select(_, Not(:precision_curve, :precision_noise)) |> 
    CSV.write(joinpath(data_dir, "network_arch_comparison.csv"), _)


##

@pipe df |> 
@subset(_, :neuron .== "all") |> 
@subset(_, :activation .== "PReLU") |> 
@subset(_, :bias .== "False") |> 
# @subset(_, :embed .== 10) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
groupby(_, [:window, :embed, :hiddendim, :layers]) |> 
combine(_, :mi => mean => :mi, :mi=>std=>:mi_sd, :precision=>mean=>:precision) |> 
@transform(_, :milo = :mi .- :mi_sd, :mihi = :mi .+ :mi_sd) |> 
@transform(_, :window = log10.(:window)) |> 
# @transform(_, :window = string.(round.(Int, log10.(:window)))) |> 
(
AlgebraOfGraphics.data(_) *
(mapping(:window=>"Window length (ms)", :mi=>"I(X,Y) (bits/s)", 
    row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, color=:embed, dodge_x=:embed=>nonnumeric) * visual(Scatter) +
    # row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, color=:embed) * visual(Scatter) +
mapping(:window=>"Window length (ms)", :milo, :mihi,
    row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, color=:embed, dodge_x=:embed=>nonnumeric) * visual(Rangebars))
    # row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, color=:embed) * visual(Rangebars))
) |> 
draw(_, scales(DodgeX = (; width = 0.1)))#, axis=(; xscale=log10))


##---------------- Activation, bias comparison figure
CairoMakie.activate!()

dt = @pipe df |> 
@subset(_, :embed .== 8) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
groupby(_, [:window, :embed, :hiddendim, :layers, :bias, :activation, :neuron]) |> 
combine(_, :mi => mean => :mi, :precision=>mean=>:precision)

# All neurons
f_all = @pipe dt |> 
@subset(_, :neuron .== "all") |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window=>"Window length (ms)", :mi=>"I(X,Y) (bits/s)", 
    row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, 
    color=:bias=>"1st layer bias", marker=:activation=>"Activation function") * 
visual(Scatter)
)

# Single neuron
f_single = @pipe dt |> 
@subset(_, :neuron .== "neuron") |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window=>"Window length (ms)", :mi=>"I(X,Y) (bits/s)", 
    row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, 
    color=:bias=>"1st layer bias", marker=:activation=>"Activation function") * 
visual(Scatter)
)

f = Figure(size=(1600, 650))
draw!(f[1,1], f_all, axis=(; xscale=log10, ))
figuregrid = draw!(f[1,2], f_single, axis=(; xscale=log10))
legend!(f[1,3], figuregrid)

for (letter_label, neuron_label, thisf) in zip(["A", "B"], ["All neurons", "Single neuron"], [f[1,1], f[1,2]])
    layout = contents(thisf)[1]
    Label(layout[1, 1, TopLeft()], letter_label,
        fontsize = 26,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
    Label(layout[0,:], neuron_label, 
        fontsize = 20, font=:bold, halign=:center)
    rowgap!(layout, Relative(0.02))
    colgap!(layout, Relative(0.01))
end

save(joinpath(fig_dir, "network_comparison", "activation_and_bias.pdf"), f)

f
GLMakie.activate!()

## Does activation function matter?
using HypothesisTests

dt = @pipe df |> 
    @transform(_, :mi = :mi ./ :window) |> 
    groupby(_, [:neuron, :hiddendim, :window, :layers, :embed, :bias, :rep, :activation]) |> 
    combine(_, :mi => mean => :mi) |> 
    unstack(_, [:neuron, :hiddendim, :window, :layers, :embed, :bias, :rep], :activation, :mi) |> 
    @transform(_, :act_dif = :LeakyReLU .- :PReLU) |> 
    disallowmissing(_)

f, ax, hs = hist(dt.act_dif, normalization=:pdf)

OneSampleTTest(dt.LeakyReLU, dt.PReLU)

dt = @pipe df |> 
    @transform(_, :mi = :mi ./ :window) |> 
    @transform(_, :window = :window .* 1000) |> 
    @subset(_, :window .< 100) |> 
    @subset(_, :bias .== "False") |> 
    groupby(_, [:neuron, :hiddendim, :window, :layers, :embed, :bias, :rep, :activation]) |> 
    combine(_, :mi => mean => :mi) |> 
    unstack(_, [:neuron, :hiddendim, :window, :layers, :embed, :bias, :rep], :activation, :mi) |>
    disallowmissing(_)

mean(dt.LeakyReLU - dt.PReLU)


##
windowlens = sort(unique(df.window))

@pipe df |> 
# @subset(_, :neuron .== "all") |> 
@subset(_, :embed .== 12) |> 
@subset(_, :window .== windowlens[2]) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
groupby(_, [:window, :embed, :hiddendim, :layers, :bias, :activation, :neuron]) |> 
combine(_, :mi => mean => :mi, :precision=>mean=>:precision) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:hiddendim, :mi=>"I(X,Y) (bits/s)", 
    row=:neuron, col=:layers=>nonnumeric, 
    color=:bias, marker=:activation) * 
visual(Scatter)
) |> 
draw(_, axis=(; xscale=log2), facet=(; linkyaxes=:rowwise))

## Embedding dim comparison

@pipe df |> 
@subset(_, :neuron .== "all") |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = round.(:window .* 1000)) |> 
groupby(_, [:window, :embed, :hiddendim, :layers]) |> 
combine(_, :mi => mean => :mi) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:embed, :mi=>"I(X,Y) (bits/s)", row=:hiddendim=>nonnumeric, col=:window=>nonnumeric, color=:layers) * 
visual(Scatter)
) |> 
draw(_, axis=(; xticks=[2,6,10,14]))

## Number of parameters

@pipe df |> 
@subset(_, :neuron .== "neuron") |> 
@subset(_, :embed .> 6) |> 
# @transform(_, :mi = :mi ./ :window) |> s
@transform(_, :n_params = :hiddendim .* :layers) |> 
@transform(_, :window = round.(:window .* 1000)) |> 
groupby(_, [:window, :layers, :n_params]) |> 
combine(_, :mi => mean => :mi) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:n_params, :mi=>"I(X,Y) (bits/s)", col=:window=>nonnumeric, color=:layers) * 
visual(Scatter)
) |> 
draw(_, axis=(; xscale=log2))

##

@pipe df |> 
@subset(_, :neuron .== "neuron") |> 
# @subset(_, :embed .== 10) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
groupby(_, [:window, :embed, :hiddendim, :layers]) |> 
combine(_, :mi => mean => :mi) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window=>"Window length (ms)", :mi=>"I(X,Y) (bits/s)", row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, color=:embed) * 
visual(Scatter)
) |> 
draw(_, axis=(; xscale=log10))

## Precision

@pipe df |> 
@subset(_, :neuron .== "all") |> 
# @subset(_, :embed .== 10) |> 
@transform(_, :window = :window .* 1000) |> 
# @subset(_, :window .< 60) |> 
@transform(_, :n_params = :hiddendim .* :layers) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window, :precision=>"Precision (ms)", color=:embed) * 
visual(Scatter)
) |> 
draw(_)#, axis=(; xscale=log10))

##
dt = @pipe df |> 
@subset(_, :embed .== 8) |> 
@subset(_, :layers .== 4) |> 
# @subset(_, :activation .== "LeakyReLU") |> 
@transform(_, :n_params = :hiddendim .* :layers) |> 
@subset(_, :neuron .== "neuron")

coldict = Dict("True" => 1, "False" => 2)

f = Figure()
for (i,gdf) in enumerate(groupby(dt, :window, sort=true))
    for (j,ggdf) in enumerate(groupby(gdf, :hiddendim))
        titlestr = string(round(ggdf.window[1] * 1000)) * " " * string(round(ggdf.hiddendim[1]))
        ax = Axis(f[j,i], xscale=log10, title=titlestr)
        for row in eachrow(ggdf)
            curve = row.precision_curve
            lines!(ax, row.precision_noise[2:end], curve[3:end] ./ row.window, 
            # lines!(ax, row.precision_noise[2:end], curve[3:end], 
                color=Makie.wong_colors()[coldict[row.bias]])
        end
        ylims!(ax, 0, maximum(dt.mi ./ dt.window))
        # ylims!(ax, 0, maximum(dt.mi))
        vlines!(ax, ggdf.window[1] * 1000, color="black", linestyle=:dash)
    end
end
Label(f[end+1,1:end], "Rounding level (ms)")
Label(f[:,0], "I(X,Y) (bits/s)", rotation=pi/2)
f

##

dt = @pipe df |> 
    @subset(_, :window .< 0.1) |> 
    @subset(_, :neuron .== "neuron") |> 
    groupby(_, [:neuron, :layers, :hiddendim, :embed, :window]) |> 
    collect(_) |> rand(_)

f = Figure()
ax = Axis(f[1,1], xscale=log10)
for row in eachrow(dt)
    curve = vec(mean(row.precision_curve, dims=1))
    lines!(ax, row.precision_noise[2:end], curve[2:end])
end
ylims!(ax, 0, nothing)
f

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