## Supplement plot for direction estimation
using Polynomials

# timing_info_colors = [colorant"#e66101", colorant"#5e3c99"]
color_dict = Dict("descending" => Makie.wong_colors()[1], "ascending" => Makie.wong_colors()[2], "uncertain" => Makie.wong_colors()[4])

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


# Read mean waveforms from 2025-03-12-1 for figure examples
mean_waves = npzread(joinpath(data_dir, "..", "2025-03-12_1", "mean_waveforms.npz"))


f = Figure(size=(1000, 500))
g_diagram = f[1,1] = GridLayout() # Placeholder for diagram
ga = f[1,2] = GridLayout() # Diagram
gb = f[1,3] = GridLayout() # cross-corr example
gc = f[1,4] = GridLayout() # time lag
gd = f[1,5] = GridLayout() # Histogram of dx/dt
ge = f[1,6] = GridLayout() # Example bar
gf = f[2,:] = GridLayout()

colsize!(f.layout, 1, Relative(0.1))

# Example mean waveforms
ax_wave = Axis(ga[1,1])
unit = "23"
inds = 40:80
# inds = 45:75
hlines!(ax_wave, [0, 1.1], color=:black)
scatterlines!(ax_wave, mean_waves[unit][inds,2] ./ maximum(mean_waves[unit][inds,2]), color=1, colorrange=(1,2.3), colormap=:viridis)
scatterlines!(ax_wave, mean_waves[unit][inds,23] ./ maximum(mean_waves[unit][inds,23]) .+ 1.1, color=2, colorrange=(1,2.3), colormap=:viridis)
hidedecorations!(ax_wave)
hidespines!(ax_wave)

# Example of X-correlation
ax_xcor_examp = Axis(gb[1,1], xlabel="Time (ms)")
inds = 45:75
maxind = findmax(mean_waves[unit][inds,2])[2]
timevec = collect((1:length(inds)) ./ 30000 .* 1000)
timevec = timevec .- timevec[maxind]
lines!(ax_xcor_examp, timevec, mean_waves[unit][inds,2] ./ maximum(mean_waves[unit][inds,2]), color=1, colorrange=(1,3), colormap=:viridis)
yvec = mean_waves[unit][inds,23] ./ maximum(mean_waves[unit][inds,23])
lines!(ax_xcor_examp, timevec, yvec, color=2, colorrange=(1,2.3), colormap=:viridis)
for i in -2:2
    lines!(ax_xcor_examp, timevec .- 0.1 * i, yvec, color=2, colorrange=(1,2.3), colormap=:viridis, linestyle=:dot, linewidth=0.9)
end
xlims!(ax_xcor_examp, -0.5, 0.5)

hideydecorations!(ax_xcor_examp)
hidespines!(ax_xcor_examp, :l, :t, :r)

# Cross-correlation values with top choice illustrated
ax_xcor = Axis(gc[1,1], xlabel="Time lag (ms)", ylabel="Cross correlation")
lags = collect(-15:15)
xcor = crosscor(mean_waves[unit][:,2], mean_waves[unit][:,23], lags, demean=false)
stem!(ax_xcor, lags, xcor)
fit_inds = argmax(xcor) .+ collect(-2:2)
pfit = Polynomials.fit(lags[fit_inds], xcor[fit_inds], 2)
polyrange = LinRange(lags[fit_inds[1]], lags[fit_inds[end]], 100)
# vertical lines for Δt
maxval, maxind = findmax(pfit.(polyrange)) 
lines!(ax_xcor, repeat([polyrange[maxind]], 2), [maxval, -0.3], color=:black)
lines!(ax_xcor, [0,0], [0, -0.3], color=:black)
text!(ax_xcor, 1.5, -0.2, text="Δt", align=(:center, :center), fontsize=18)
# Highlight for polyfit
scatter!(ax_xcor, lags[fit_inds], xcor[fit_inds], color=:grey, markersize=12)
lines!(ax_xcor, polyrange, pfit.(polyrange), color=(:grey, 0.5), linewidth=3)

xlims!(ax_xcor, -7, 7)

# Histogram of dx/dts 
thisdf = @pipe df_vels  |> 
    @subset(_, :unit .== 23, :moth .== "2025-03-12-1") |> 
    @transform(_, :vel = :dx ./ :dt)
ax_hist = Axis(gd[1,1], xlabel="Δx/Δt (m/s)", ylabel="Count", xticks=[-1, 0, 1])
vlines!(ax_hist, 0, color=:black)
bins = range(minimum(thisdf.vel), maximum(thisdf.vel), 20)
hist!(ax_hist, thisdf[thisdf.vel .< 0, :vel], bins=bins, color=Makie.wong_colors()[1])
hist!(ax_hist, thisdf[thisdf.vel .> 0, :vel], bins=bins, color=Makie.wong_colors()[2])

arr_height = 15
arrows2d!(ax_hist, Point2f(0, arr_height), [[-0.5,0]], 
    minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5,
    color=Makie.wong_colors()[1])
arrows2d!(ax_hist, Point2f(0, arr_height), [[+0.5,0]], 
    minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5,
    color=Makie.wong_colors()[2])
text!(ax_hist, 0.1, 16.5, text="Ascend", color=Makie.wong_colors()[2], align=(:left, :center), fontsize=14)
text!(ax_hist, -0.1, 16.5, text="Descend", color=Makie.wong_colors()[1], align=(:right, :center), fontsize=14)

xlims!(ax_hist, -1.5, 1.5)
ylims!(ax_hist, 0, 18)

# Example bar for this example neuron
thisdf = @pipe leftjoin(dfdir, df_ndir, on=[:moth, :globalunit]) |> 
    @subset(_, :globalunit .== 253)
ax_bar_ex = Axis(ge[1,1], ylabel="Frac. of velocity estimates")
barplot!(ax_bar_ex, [1,1], vcat(thisdf.n_descend ./ thisdf.n_total, thisdf.n_ascend ./ thisdf.n_total),
    stack=[1,2],
    color=[1,2],
    colormap=[Makie.wong_colors()[1], Makie.wong_colors()[2]]
)
thiscolor = Makie.wong_colors()[2]
hsl = convert(HSL, RGB(thiscolor.r, thiscolor.g, thiscolor.b))
final_color = HSL(hsl.h, hsl.s, hsl.l * 0.8)
lines!(ax_bar_ex, [1,1], [thisdf.HDIlo[1], thisdf.HDIup[1]], color=final_color)
hlines!(ax_bar_ex, 0.5, color=:black, linestyle=:dash)

hidexdecorations!(ax_bar_ex)
hidespines!(ax_bar_ex, :b, :r, :t)
ylims!(ax_bar_ex, 0, 1)

colsize!(f.layout, 6, Relative(0.05))

# Barplots 
ax = Axis(gf[1,1], ylabel="Frac. of velocity estimates")
order_map = Dict("ascending" => 1, "uncertain" => 2, "descending" => 3)
sort!(df_direct, [order(:direction, by = x -> order_map[x]), order(:mean)])
barplot!(ax, 
    repeat(1:nrow(df_direct), 2), 
    vcat(df_direct.n_descend ./ df_direct.n_total, df_direct.n_ascend ./ df_direct.n_total),
    stack=vcat(ones(Int, nrow(df_direct)), 2 .* ones(Int, 1:nrow(df_direct))),
    color=vcat(ones(nrow(df_direct)), 2 .* ones(1:nrow(df_direct))),
    gap=0,
    colormap=[Makie.wong_colors()[1], Makie.wong_colors()[2]]
)
for (i,row) in enumerate(eachrow(df_direct))
    # Darken if ascending or descending, otherwise just use pink
    if row.direction == "uncertain"
        final_color = Makie.wong_colors()[4]
    else
        thiscolor = color_dict[row.direction]
        hsl = convert(HSL, RGB(thiscolor.r, thiscolor.g, thiscolor.b))
        final_color = HSL(hsl.h, hsl.s, hsl.l * 0.8)
    end
    # thiscol = row.direction == "uncertain" ? Makie.wong_colors()[4] : :black
    lines!(ax, [i,i], [row.HDIlo, row.HDIup], color=final_color)
end
hlines!(ax, 0.5, color=:black, linestyle=:dash)
limits!(ax, (0.5, 144.5), (0, 1))
hidexdecorations!(ax, label=false)

apply_letter_label(g_diagram, "A")
apply_letter_label(gb, "B")
apply_letter_label(gc, "C")
apply_letter_label(gd, "D")
apply_letter_label(ge, "E")
apply_letter_label(gf, "F")

display(f)


save(joinpath(fig_dir, "fig_supp_direction_estimation.pdf"), f)