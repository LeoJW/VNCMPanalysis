using HDF5
using Statistics
using GLMakie
using AlgebraOfGraphics
using DataFrames
using DataFramesMeta
using Pipe


analysis_dir = @__DIR__
data_dir = joinpath(analysis_dir, "..", "localdata", "estimation_runs")

file = "network_arch_comparison_PACE_2025-06-05.h5"

precision_curves = h5read(joinpath(data_dir, file), "dict_1")
time_per_epoch = h5read(joinpath(data_dir, file), "dict_2")

# Construct dataframe
first_row = vcat("neuron", split(first(keys(time_per_epoch)), "_"))
# first_row = split(first(keys(time_per_epoch)), "_")
names = vcat(first_row[1:2:end], ["time", "mi"])
# is_numeric = vcat([tryparse(Float64, x) !== nothing for x in first_row[2:2:end]], true, true)
is_numeric = [false, true, true, true, false, true, true, true]
types = [x ? Float64 : String for x in is_numeric]

df = DataFrame(Dict(names[i] => types[i][] for i in eachindex(names)))
df = df[!, Symbol.(names)] # Undo name sorting
for key in keys(time_per_epoch)
    keysplit = vcat("neuron", split(key, "_"))[2:2:end]
    # keysplit = split(key, "_")[2:2:end]
    vals = map(x->(return is_numeric[x[1]] ? parse(Float64, x[2]) : x[2]), enumerate(keysplit))
    push!(df, vcat(vals, time_per_epoch[key], mean(precision_curves[key], dims=1)[1]))
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
@subset(_, :neuron .== "single") |> 
@transform(_, :layout = ifelse.(:layout .== "None", "0", :layout)) |> 
@subset(_, :stride .== 2) |> 
(
AlgebraOfGraphics.data(_) *
mapping(:mi, :time, col=:layout, color=:layers=>nonnumeric, row=:filters=>nonnumeric) * 
visual(Scatter)
) |> 
draw(_)
