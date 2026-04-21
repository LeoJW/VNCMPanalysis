
## Supplement plot for direction estimation

dfdir = @pipe CSV.read(joinpath(data_dir, "..", "direction_estimate_stats_all_units.csv"), DataFrame) |> 
    rename(_, :unit => :neuron, Symbol("%>comp") => :prob_descend, Symbol("%<comp") => :prob_ascend) |> 
    transform!(_, [:HDIlo, :HDIup] =>
        ByRow((HDIlo, HDIup) -> 
            HDIlo < 0.5 && HDIup < 0.5 ? "ascending" :
            HDIlo > 0.5 && HDIup > 0.5 ? "descending" :
            "uncertain") =>
        :direction
    ) |> 
    transform!(_, :moth => ByRow(x -> replace(x, r"_1$" => "-1")) => :moth)
df_vels = @pipe CSV.read(joinpath(data_dir, "..", "velocity_estimates_all_units.csv"), DataFrame) |> 
    @transform!(_, :vel = :dx ./ :dt) |> 
    transform!(_, :moth => ByRow(x -> replace(x, r"_1$" => "-1")) => :moth)
df_ndir = @pipe df_vels |> 
    groupby(_, [:moth, :unit, :globalunit]) |> 
    combine(_, 
        :vel => (v -> sum(v .< 0)) => :n_descend,
        :vel => (v -> sum(v .> 0)) => :n_ascend,
        :vel => length => :n_total
    )
df_direct = leftjoin(dfdir, df_ndir, on=[:moth, :globalunit])

# Small copy of df just to pick out only neurons meeting 1000 spike condition
smalldf = @pipe df |> 
    @subset(_, :muscle .== "all", :peak_mi, :nspikes .> 1000) |> 
    select(_, [:moth, :neuron, :nspikes])
df_direct = leftjoin(df_direct, smalldf, on=[:moth, :neuron])
@subset!(df_direct, (!).(ismissing.(:nspikes)))



f = Figure()
ga = f[1,1] = GridLayout() # Diagram
gb = f[1,2] = GridLayout() # cross-corr example
gc = f[1,3] = GridLayout() # time lag
gd = f[1,4] = GridLayout() # Histogram of dx/dt
ge = f[1,5] = GridLayout() # Example bar
gf = f[2,:] = GridLayout()

ax = Axis(gf[1,1])

sort!(df_direct, order(:mean))

barplot!(ax, 
    repeat(1:nrow(df_direct), 2), 
    vcat(df_direct.n_descend ./ df_direct.n_total, df_direct.n_ascend ./ df_direct.n_total),
    stack=vcat(ones(Int, nrow(df_direct)), 2 .* ones(Int, 1:nrow(df_direct))),
    color=vcat(ones(nrow(df_direct)), 2 .* ones(1:nrow(df_direct))),
    gap=0
)


apply_letter_label(gf, "A")
f
