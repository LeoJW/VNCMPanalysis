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
    curve = savitzky_golay(vec(mean(mi[:,2:end], dims=1)), sg_window, 2; deriv=0).y ./ mi[1,1]

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
            find_precision_threshold(precision_noise[key][2:end] .* 1000, precision_curves[key][3:end]),
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
# Extra larger network runs
for task in 0:5
    read_network_arch_file!(df, joinpath(data_dir, "2025-06-23_network_comparison_PACE_task_$(task).h5"), task)
end


## What really did ISI do?
dfisi = DataFrame()
for task in 0:5
    read_network_arch_file!(dfisi, joinpath(data_dir, "2025-06-20_network_comparison_PACE_task_$(task)_hour_14.h5"), task)
end

@pipe dfisi |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window, :precision, col=:neuron, color=:ISI) *
visual(Scatter)
) |> 
draw(_)
##
coldict = Dict("True" => Makie.wong_colors()[1], "False" => Makie.wong_colors()[2])

rows = eachrow(@subset(dfisi, :window .> 0.4))
f = Figure()
ax = Axis(f[1,1], xscale=log10)
for row in rows
    lines!(ax, row.precision_noise[2:end], row.precision_curve[3:end],
        color=coldict[row.ISI])
end
f


##
GLMakie.activate!()

@pipe df |> 
@subset(_, :neuron .== "all") |> 
@subset(_, :activation .== "PReLU") |> 
# @subset(_, :activation .== "LeakyReLU") |> 
@subset(_, :bias .== "False") |> 
# @subset(_, :embed .== 10) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
groupby(_, [:window, :embed, :hiddendim, :layers]) |> 
combine(_, :mi => mean => :mi, :mi=>std=>:mi_sd, :precision=>mean=>:precision) |> 
@transform(_, :milo = :mi .- :mi_sd, :mihi = :mi .+ :mi_sd) |> 
@transform(_, :window = log10.(:window)) |> 
(
AlgebraOfGraphics.data(_) *
(mapping(:window=>"Window length (ms)", :mi=>"I(X,Y) (bits/s)", 
    row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, color=:embed, dodge_x=:embed=>nonnumeric) * visual(Scatter) +
    # row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, color=:embed) * visual(Scatter) +
mapping(:window=>"Window length (ms)", :milo, :mihi,
    row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, color=:embed, dodge_x=:embed=>nonnumeric) * visual(Rangebars))
    # row=:hiddendim=>nonnumeric, col=:layers=>nonnumeric, color=:embed) * visual(Rangebars))
) |> 
draw(_, scales(DodgeX = (; width = 0.1)), axis=(; limits=(nothing, (0, maximum(df.mi ./ df.window)))))

## How does precision vary?

dt = @pipe df |> 
@subset(_, :neuron .== "neuron") |> 
# @subset(_, :activation .== "PReLU") |> 
@subset(_, :activation .== "LeakyReLU") |> 
@subset(_, :bias .== "False") |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
groupby(_, [:window, :embed, :hiddendim, :layers]) |> 
combine(_, :mi => mean => :mi, :mi=>std=>:mi_sd, :precision=>mean=>:precision) |> 
@subset(_, :mi .> 0) |> 
@transform(_, :window = log10.(:window))

@pipe dt[sortperm(dt.precision),:] |> 
(
AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, 
    row=:embed=>nonnumeric, 
    col=:layers=>nonnumeric, color=:hiddendim=>log2, group=:hiddendim=>nonnumeric) * 
    (visual(Scatter) + visual(Lines))
) |> 
draw(_, axis=(; yscale=log10))

##
@pipe df |> 
@subset(_, :layers .== 4, :bias .== "False") |> 
groupby(_, [:neuron, :hiddendim, :activation, :window, :embed]) |> 
combine(_, :time => mean => :time, :mi => mean => :mi) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:embed, :mi, color=:window, row=:neuron, col=:hiddendim=>nonnumeric) * visual(Scatter)
) |> 
draw(_)


## At a given window length, how does hiddendim change MI?

GLMakie.activate!()

@pipe df |> 
@subset(_, :activation .== "PReLU") |> 
@subset(_, :window .== sort(unique(df.window))[2]) |> 
# @subset(_, :activation .== "LeakyReLU") |> 
@subset(_, :bias .== "False") |> 
# @subset(_, :embed .== 10) |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = :window .* 1000) |> 
groupby(_, [:window, :embed, :hiddendim, :layers, :neuron]) |> 
combine(_, :mi => mean => :mi, :mi=>std=>:mi_sd, :precision=>mean=>:precision) |> 
@transform(_, :milo = :mi .- :mi_sd, :mihi = :mi .+ :mi_sd) |> 
@transform(_, :hiddendim = log2.(:hiddendim)) |> 
(
AlgebraOfGraphics.data(_) *
(mapping(:hiddendim=>"Hidden dimension", :mi=>"I(X,Y) (bits/s)", 
    row=:neuron, col=:layers=>nonnumeric, color=:embed, dodge_x=:embed=>nonnumeric) * visual(Scatter) +
mapping(:hiddendim=>"Hidden dimension", :milo, :mihi,
    row=:neuron, col=:layers=>nonnumeric, color=:embed, dodge_x=:embed=>nonnumeric) * visual(Rangebars))
) |> 
draw(_, scales(DodgeX = (; width = 0.1)), facet=(; linkyaxes=:minimal))




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
@subset(_, :activation .== "PReLU") |> 
@subset(_, :bias .== "False") |>  
@transform(_, :n_params = :hiddendim .* :layers) |> 
@subset(_, :neuron .== "all")

coldict = Dict(hd => i for (i,hd) in enumerate(sort(unique(df.hiddendim))))

f = Figure()
for (i,gdf) in enumerate(groupby(dt, :window, sort=true))
    titlestr = string(round(gdf.window[1] * 1000))
    ax = Axis(f[1,i], xscale=log10, title=titlestr)
    for (j,ggdf) in enumerate(groupby(gdf, :hiddendim))
        for row in eachrow(ggdf)
            curve = row.precision_curve
            lines!(ax, row.precision_noise[2:end], curve[3:end] ./ row.window, 
            # lines!(ax, row.precision_noise[2:end], curve[3:end], 
                color=row.hiddendim, colorrange=extrema(df.hiddendim))
        end
        ylims!(ax, 0, maximum(dt.mi ./ dt.window))
        # ylims!(ax, 0, maximum(dt.mi))
        vlines!(ax, ggdf.window[1] * 1000, color="black", linestyle=:dash)
    end
end
Label(f[end+1,1:end], "Rounding level (ms)")
Label(f[:,0], "I(X,Y) (bits/s)", rotation=pi/2)
f

## Let's come up with a better precision change point detection method
using GLM

row = rand(eachrow(df))

f = Figure()
ax = Axis(f[1,1], xscale=log10)
scatterlines!(ax, row.precision_noise, row.precision_curve[2:end] .- row.precision_curve[2])

# Page Hinkley CUMSUM
# curve = row.precision_curve[2:end] ./ row.precision_curve[2]
# for (i,ω) in enumerate(logrange(0.001, 10, 10)) 
#     S = zero(curve)
#     for i in 1:length(curve)-1
#         S[i+1] = max(0, S[i] - curve[i] - ω)
#     end
#     scatterlines!(ax, row.precision_noise, S, color=log.(ω), colorrange = log.([0.001, 10]))
# end

# Sliding linear regression
function sliding_slope(y::Vector{Float64}, x::Vector{Float64}; window=12)
    slopes = zeros(length(x))
    for i in 1:(length(x) - window)
        mod = lm(@formula(y ~ x), DataFrame(x=x[i:i+window], y=y[i:i+window]))
        slopes[i] = coef(mod)[2]
    end
    return slopes
end
curve = row.precision_curve[2:end] ./ row.precision_curve[2]
slopes = sliding_slope(curve, log.(row.precision_noise); window=20)
scatterlines!(ax, row.precision_noise, slopes)
ind = findfirst(slopes .< -0.1)
vlines!(ax, row.precision_noise[ind])

# Dumb 90% threshold
curve = row.precision_curve[2:end] ./ row.precision_curve[2]
ind = findfirst(curve .< 0.9)
vlines!(ax, row.precision_noise[ind], color=:black, linestyle=:dash)

ylims!(ax, extrema(row.precision_curve[2:end] .- row.precision_curve[2]) .+ [0, 0.1])
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

## ------------------------------------ TIme shifting test ------------------------------------

function read_timeshift_file!(df, file)
    precision_noise = h5read(joinpath(data_dir, file), "dict_0")
    precision_curves = h5read(joinpath(data_dir, file), "dict_1")
    # Construct dataframe
    first_row = split(first(keys(precision_noise)), "_")
    names = vcat(first_row[1:2:end], ["mi", "precision", "precision_curve", "precision_noise"])
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat([x ? Float64 : String for x in is_numeric], Float64, Float64, Vector{Float64}, Vector{Float64})
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(precision_noise)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        # vals[findfirst(names .== "rep")] = task
        push!(thisdf, vcat(
            vals,
            precision_curves[key][1] .* log2(exp(1)), 
            find_precision_threshold(precision_noise[key] .* 1000, precision_curves[key][2:end]),
            [precision_curves[key] .* log2(exp(1))],
            [precision_noise[key] .* 1000]
        ))
    end
    append!(df, thisdf)
end

df = DataFrame()
read_timeshift_file!(df, joinpath(data_dir, "2025-07-04_timeshift_test_PACE_hour_15.h5"))

##
@pipe df[sortperm(df.shift),:] |> 
@transform(_, :mi = :mi ./ :window) |> 
@transform(_, :window = round.(:window, digits=2)) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:shift, :mi, row=:window=>nonnumeric) * visual(ScatterLines)
) |> 
draw(_)


## ------------------------------------ Kinematics choosing embedding dim ------------------------------------

function read_kinematics_embed_file!(df, file)
    mi = h5read(joinpath(data_dir, file), "dict_0")
    # Construct dataframe
    first_row = split(first(keys(mi)), "_")
    names = vcat(first_row[1:2:end], "mi")
    is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
    types = vcat([x ? Float64 : String for x in is_numeric], Float64)
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(mi)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        push!(thisdf, vcat(vals, mi[key] .* log2(exp(1))))
    end
    append!(df, thisdf)
end

df = DataFrame()
for i in 0:3
    read_kinematics_embed_file!(df, joinpath(data_dir, "2025-07-09_kinematics_embed_PACE_task_$(i).h5"))
end
muscle_names_dict = Dict(
    "lax-lba-lsa-rsa-rba-rax" => "steering",
    "lax-lba-lsa" => "Lsteering",
    "rax-rba-rsa" => "Rsteering",
    "ldvm-ldlm-rdlm-rdvm" => "power",
    "ldvm-ldlm" => "Lpower",
    "rdvm-rdlm" => "Rpower"
)
for muscle in single_muscles
    muscle_names_dict[muscle] = muscle
end
df = @pipe df |> 
    @transform(_, :single = ifelse.(occursin.("-", :muscle), false, true)) |> 
    @transform(_, :muscle = getindex.(Ref(muscle_names_dict), :muscle))
##

@pipe df[sortperm(df.embed),:] |> 
@transform(_, :mi = :mi ./ :window) |> 
groupby(_, [:muscle, :moth, :embed, :window]) |> 
combine(_, 
    :mi => mean => :mi, 
    :mi => (x->mean(x) - std(x)) => :mi_lo,
    :mi => (x->mean(x) + std(x)) => :mi_hi) |> 
@subset(_, :mi .> 0) |> 
# groupby(_, [:muscle, :moth, :window]) |> 
# transform(_, [:mi, :embed] => ((mi, embed) -> mi .- first(mi[embed .== 4])) => :mi) |> 
(
AlgebraOfGraphics.data(_) * (
mapping(:embed, :mi, dodge_x=:window=>nonnumeric, color=:window, group=:window=>nonnumeric, row=:muscle, col=:moth) * 
visual(ScatterLines) + 
mapping(:embed, :mi_lo, :mi_hi, dodge_x=:window=>nonnumeric, color=:window, group=:window=>nonnumeric, row=:muscle, col=:moth) *
visual(Rangebars)
)
) |> 
draw(_, scales(DodgeX = (; width = 2)))

##

@pipe df[sortperm(df.window),:] |> 
@transform(_, :mi = :mi ./ :window) |> 
groupby(_, [:muscle, :moth, :embed, :window]) |> 
combine(_, 
    :mi => mean => :mi, 
    :mi => (x->mean(x) - std(x)) => :mi_lo,
    :mi => (x->mean(x) + std(x)) => :mi_hi) |> 
@subset(_, :mi .> 0) |> 
(
AlgebraOfGraphics.data(_) * (
mapping(:window, :mi, color=:embed, group=:embed=>nonnumeric, row=:moth=>nonnumeric, col=:muscle) * 
visual(ScatterLines)
)
) |> 
draw(_)

## ------------------------------------ Kinematics precision ------------------------------------

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

df = DataFrame()
for task in 0:5
    read_precision_kinematics_file!(df, joinpath(data_dir, "2025-07-02_kinematics_precision_PACE_task_$(task)_hour_09.h5"), task)
end

single_muscles = [
    "lax", "lba", "lsa", "ldvm", "ldlm", 
    "rdlm", "rdvm", "rsa", "rba", "rax"
]

##

@pipe df |> 
@transform(_, :neuron = ifelse.(in.(:neuron, (single_muscles,)), "single", :neuron)) |> 
(AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, color=:moth, row=:neuron) * visual(Scatter)
) |> 
draw(_, axis=(; yscale=log10))

## MI vs precision

@pipe df |> 
@transform(_, :mi = :mi ./ :window) |> 
@subset(_, in.(:neuron, (single_muscles,))) |> 
transform!(_, :neuron =>  ByRow(s ->s[2:end]) => :neuron) |> 
(AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, color=:window, col=:moth, row=:neuron) * visual(Scatter)
) |> 
draw(_, axis=(; yscale=log10))

##

dt = @pipe df |> 
@transform(_, :mi = :mi ./ :window) |> 
# @subset(_, in.(:neuron, (single_muscles,))) |> 
transform!(_, :neuron =>  ByRow(s ->s[2:end]) => :neuron) |> 
groupby(_, [:window, :moth, :neuron]) |> 
combine(_, :mi => mean => :mi, :precision => mean => :precision)
@pipe dt[sortperm(dt.precision),:] |> 
(AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, color=:window, col=:moth, row=:neuron) * visual(ScatterLines)
) |> 
draw(_, axis=(; yscale=log10))

## Effect of window size on precision; Depends on how many muscles used!

dt = @pipe df |> 
@transform(_, :mi = :mi ./ :window) |> 
groupby(_, [:window, :neuron, :moth]) |> 
combine(_, :mi => mean => :mi, :precision => mean => :precision) |> 
groupby(_, [:neuron, :moth]) |> 
@transform(_, :groupcol = string.(:neuron) .* string.(:moth)) |> 
@transform(_, :single = in.(:neuron, (single_muscles,))) 
# @transform(_, :neuron = ifelse.(:single, "single", :neuron))

@pipe dt[sortperm(dt.window),:] |> 
(AlgebraOfGraphics.data(_) *
mapping(:window, :mi, color=:mi, row=:moth, col=:neuron, group=:groupcol) * visual(ScatterLines)
) |> 
draw(_, axis=(; xscale=log10))#, yscale=log10))


##

dt = @pipe df |> 
@subset(_, :rep .== 0) |> 
@transform(_, :single = in.(:neuron, (single_muscles,))) |> 
# @transform(_, :neuron = ifelse.(:single, "single", :neuron)) |> 
@subset(_, (!).(:single)) |> 
flatten(_, [:subset, :mi_subset]) |> 
@transform(_, :mi_subset = :mi_subset ./ :window, :mi = :mi ./ :window) |> 
@groupby(_, [:subset, :window, :moth, :neuron]) |> 
combine(_, :mi_subset => mean => :mi_subset, :mi_subset => std => :mi_subset_std, :mi => first => :mi) |> 
@groupby(_, [:window, :moth, :neuron]) |> 
combine(_) do d
    row = copy(DataFrame(d[1,:]))
    row.mi_subset .= row.mi
    row.subset .= 1
    return vcat(row, d)
end |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:subset, :mi_subset, color=:window=>log, col=:moth, row=:neuron, group=:window=>nonnumeric) * 
visual(ScatterLines)
) |> 
draw(_)

##
f = Figure()
ax = Axis(f[1,1])
row = rand(eachrow(@subset(df, :rep .== 0)))
scatterlines!(ax, 
    vcat(1, sort(unique(row.subset))), 
    vcat(row.mi, [mean(row.mi_subset[row.subset .== v]) for v in sort(unique(row.subset))])
)
f

## ------------------------------------ Subsampling results ------------------------------------

file = "subsampling_PACE_2025-06-06.h5"
#subsamples = h5read(joinpath(data_dir, file), "dict_0")
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

##

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

function read_run_file!(df, file, task)
    precision_levels = h5read(joinpath(data_dir, file), "dict_0")
    precision_curves = h5read(joinpath(data_dir, file), "dict_1")
    precision_noise_curves = h5read(joinpath(data_dir, file), "dict_2")
    # Construct dataframe
    first_row = split(first(keys(precision_levels)), "_")
    not_has_noise = typeof(first(values(precision_noise_curves))) == Vector{String}
    if !not_has_noise # Onlder runs without noise precision
        names = vcat(first_row[1:2:end], ["mi", "precision", "precision_noise", "precision_curve", "precision_levels"])
        is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
        types = vcat(
            [x ? Float64 : String for x in is_numeric], 
            Float64, Float64, Float64, Vector{Float64}, Vector{Float64}
        )
    else
        names = vcat(first_row[1:2:end], ["mi", "precision", "precision_curve", "precision_levels"])
        is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]])
        types = vcat(
            [x ? Float64 : String for x in is_numeric], 
            Float64, Float64, Vector{Float64}, Vector{Float64}
        )
    end
    thisdf = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
    thisdf = thisdf[!, Symbol.(names)] # Undo name sorting
    for key in keys(precision_levels)
        keysplit = split(key, "_")[2:2:end]
        vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
        if !not_has_noise
            push!(thisdf, vcat(
                vals,
                precision_curves[key][1] .* log2(exp(1)), 
                find_precision_threshold(precision_levels[key] .* 1000, precision_curves[key][2:end]),
                find_precision_noise_threshold(precision_levels[key] .* 1000, precision_noise_curves[key]),
                [precision_curves[key] .* log2(exp(1))],
                [precision_levels[key] .* 1000]
            ))
        else
            push!(thisdf, vcat(
                vals,
                precision_curves[key][1] .* log2(exp(1)), 
                find_precision_threshold(precision_levels[key] .* 1000, precision_curves[key][2:end]),
                [precision_curves[key] .* log2(exp(1))],
                [precision_levels[key] .* 1000]
            ))
        end
    end
    append!(df, thisdf)
end



dfisi = DataFrame()
read_run_file!(dfisi, joinpath(data_dir, "2025-07-10_quicktest_single_neurons_PACE_task_0_hour_23.h5"), 0)
# read_run_file!(dfisi, joinpath(data_dir, "2025-07-10_quicktest_single_neurons_PACE_task_0_hour_17.h5"), 0)
df = DataFrame()
for task in 0:3
    read_run_file!(df, joinpath(data_dir, "2025-07-12_quicktest_single_neurons_PACE_task_$(task).h5"), 0)
end

##

# @pipe dfisi[sortperm(dfisi.window),:] |> 
@pipe df[sortperm(df.window),:] |> 
# @subset(_, :hiddendim .< 129) |> 
@subset(_, :bs .== 1024, :neuron .== 7) |> 
@transform(_, :comp = :window .* :bs) |> 
@transform(_, :mi = :mi ./ :window) |> 
@subset(_, :mi .> 0) |> 
stack(_, [:mi, :precision, :precision_noise]) |> 
# stack(_, [:mi, :precision]) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:window, :value, row=:variable, col=:muscle, color=:hiddendim=>log2, group=:hiddendim=>nonnumeric) * visual(ScatterLines)
) |> 
draw(_, facet=(; linkyaxes=:none))

##
@pipe ddf[sortperm(ddf.window),:] |> 
@transform(_, :mi = :mi ./ :window) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:mi, :precision, color=:bs=>log2, group=:bs=>nonnumeric) * visual(ScatterLines)
) |> 
draw(_)