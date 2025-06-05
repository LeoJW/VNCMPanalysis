using HDF5
using Statistics
using GLMakie
using AlgebraOfGraphics
using DataFrames
using DataFramesMeta
using Pipe
using SavitzkyGolay

function find_precision(noise_levels, mi)
    curve = vec(mean(mi, dims=1))
    deriv = savitzky_golay(curve, 5, 2; deriv=2).y
    return noise_levels[findmin(deriv)[2]]
end


analysis_dir = @__DIR__
data_dir = joinpath(analysis_dir, "..", "localdata", "estimation_runs")

file = "network_arch_comparison_PACE_2025-06-05.h5"

precision_curves = h5read(joinpath(data_dir, file), "dict_1")
time_per_epoch = h5read(joinpath(data_dir, file), "dict_2")
noise_levels = vcat(0, logrange(period, 0.04, 20) .* 1000)

# Construct dataframe
first_row = vcat("neuron", split(first(keys(time_per_epoch)), "_"))
# first_row = split(first(keys(time_per_epoch)), "_")
names = vcat(first_row[1:2:end], ["time", "mi", "precision"])
# is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]], true, true)
is_numeric = [false, true, true, true, false, true, true, true, true]
types = [x ? Float64 : String for x in is_numeric]

df = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
df = df[!, Symbol.(names)] # Undo name sorting
for key in keys(time_per_epoch)
    keysplit = vcat("neuron", split(key, "_"))[2:2:end]
    # keysplit = split(key, "_")[2:2:end]
    vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
    push!(df, vcat(
        vals, 
        time_per_epoch[key], 
        mean(precision_curves[key], dims=1)[1], 
        find_precision(noise_levels, precision_curves[key])
    ))
end

##


@pipe df |> 
stack(_, [:mi, :time]) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:layout, :value, color=:filters=>nonnumeric, dodge=:filters=>nonnumeric, row=:layers=>nonnumeric, col=:variable) * 
visual(BoxPlot)
) |> 
draw(_, facet=(; linkyaxes=:none))

##
@pipe df |> 
@subset(_, :neuron .== "all") |> 
@transform(_, :layout = ifelse.(:layout .== "None", "0", :layout)) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:layout, :mi, color=:stride=>nonnumeric, dodge=:stride=>nonnumeric, col=:layers=>nonnumeric, row=:filters=>nonnumeric) * 
visual(BoxPlot)
) |> 
draw(_)

##

@pipe df |> 
@subset(_, :neuron .== "all") |> 
@transform(_, :layout = ifelse.(:layout .== "None", "0", :layout)) |> 
@subset(_, :stride .== 2) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:time, :mi, col=:layout, color=:layers=>nonnumeric, row=:filters=>nonnumeric, marker=:stride=>nonnumeric) * 
visual(Scatter)
) |> 
draw(_)


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