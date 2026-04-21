"""
figure_phasic_information.jl

Phasic neurons explain information figure (Figure 5?) 

This script requires running the first section of main_figures.jl and main_spike_figures.jl to get the necessary imports, dataframes, 
and other things set up and loaded. It's just put here to keep main_figures.jl from becoming 10,000 lines long
"""

GLMakie.activate!() # Cairo doesn't do circular histograms correctly

function phasic_neuron_figure()
muscle_colors = [
    "lax" => "#94D63C", "rax" => "#6A992A",
    "lba" => "#AE3FC3", "rba" => "#7D2D8C",
    "lsa" => "#FFBE24", "rsa" => "#E7AC1E",
    "ldvm"=> "#66AFE6", "rdvm"=> "#2A4A78",
    #"ldlm"=> "#E87D7A", "rdlm"=> "#C14434"
]
color_dict = Dict(
    "descending" => Makie.wong_colors()[1], 
    "ascending" => Makie.wong_colors()[2], 
    "uncertain" => Makie.wong_colors()[3])
timing_colors = Dict("No spike timing info" => "#5e3c99", "Spike timing info" => "#e66101")
example_neurons = ["76", "27", "11", "88"] # Descending order of mi, but will be plot in reverse order (makes color/draw order work)
example_neuron_colors = Dict(
    "76" => Makie.wong_colors()[1],
    "27" => Makie.wong_colors()[5],
    "11" => Makie.wong_colors()[4],
    "88" => Makie.wong_colors()[7]
)
circ_sort_rel_size = 0.37

# function phasic_neuron_figure()
f = Figure(size=(1900, 1500))

top_row = f[1,1:3] = GridLayout()

# -------- First panel, example neuron firing
ga = top_row[1, 1] = GridLayout()
ax_examp_range = [776, 778] .- 0.1
ax_examp = Axis(ga[1,1], 
    xticks=(ax_examp_range, ["0","2"]), limits=(tuple(ax_examp_range...), nothing),
    xticklabelspace = 0.0, xlabel="Time (s)"
)
wb_duration_thresholds = (32.5, 100) # ms ldlm
thismoth = replace.(moths[1], r"-1$" => "_1")
spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))

mask_neur = (spikes["76"] .> (ax_examp_range[1] - 1) * fsamp) .&& (spikes["76"] .< (ax_examp_range[2] + 1) * fsamp)
mask = (spikes["ldlm"] .> (ax_examp_range[1] - 1) * fsamp) .&& (spikes["ldlm"] .< (ax_examp_range[2] + 1) * fsamp)
spikevec = spikes["ldlm"][mask]

diffvec = diff(spikevec) ./ fsamp .* 1000 # units of ms
mask = (diffvec .> wb_duration_thresholds[1]) .&& (diffvec .< wb_duration_thresholds[2])
start_inds = spikevec[findall(vcat(false, mask))]
vspan!(ax_examp, start_inds[1:2:end-1] ./ fsamp, start_inds[2:2:end] ./ fsamp, color=RGBAf(0.9, 0.9, 0.9, 1.0))
vlines!(ax_examp, spikevec ./ fsamp, ymin=0, ymax=0.5, color=:black)
vlines!(ax_examp, spikes["76"][mask_neur] ./ fsamp, ymin=0.5, ymax=1.0, color=Makie.wong_colors()[1])
textlabel!(ax_examp, 0.95, 0.25, text="LDLM", space=:relative, fontsize=18, text_align=(:right, :center))
textlabel!(ax_examp, 0.95, 0.75, text="Neuron 76", space=:relative, fontsize=18, text_align=(:right, :center), text_color=Makie.wong_colors()[1])
hideydecorations!(ax_examp)
hidespines!(ax_examp)
apply_letter_label(ga, "A")


# -------- Second panel, circular hist with muscles and a neuron
max_mi_neuron = @pipe df |> 
    @subset(_, :peak_mi, :nspikes .> 1000, :muscle .== "all", :label .== "good") |> 
    @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi), :moth = replace.(:moth, r"-1$" => "_1")) |> 
    groupby(_, :moth) |> 
    combine(_, [:neuron, :mi] => (n,mi) -> n[findmax(mi)[2]]) |> 
    @transform(_, :neuron_mi_function = string.(round.(Int, :neuron_mi_function)))
max_mi_neuron = Dict(Pair.(max_mi_neuron.moth, max_mi_neuron.neuron_mi_function))

gb = top_row[2,1] = GridLayout(padding=(0,0,0,0), tellwidth=false, tellheight=false)
phase_dict, wblen_dict, muscle_phase_dict = get_phase_dict(thismoth)
ax_muscle_hist = PolarAxis(gb[1:end,1], 
    radius_at_origin=-1,
    clip=false, clip_r=false
)
muscle_theta_r = Dict(
    "ax" => [pi+0.17*pi, 0.87], # Angle (radians) and radius of each text placement
    "ba" => [pi*0.3, 0.7],
    "sa" => [pi*0.08, 0.8],
    "dvm"=> [2.75, 1.1]
)
rlims!(ax_muscle_hist, -1.0, 0.6)
hiderdecorations!(ax_muscle_hist)
hidespines!(ax_muscle_hist)
lines!(ax_muscle_hist, [0, 0], [-1, 0.6], color=:black, linewidth=3)
for i in eachindex(muscle_colors)
    muscle = muscle_colors[i][1]
    if !(muscle in keys(muscle_phase_dict))
        continue
    end
    hist!(ax_muscle_hist, muscle_phase_dict[muscle] .* 2*pi, 
        bins=90, normalization=:pdf, #scale_to=1.0,
        color=(muscle_colors[i][2], 0.5)
    )
    # Muscle label for one muscle of each pair
    if muscle[1] == 'r'
        # mean_angle = mean(muscle_phase_dict[muscle] .* 2*pi)
        text!(ax_muscle_hist, muscle_theta_r[muscle[2:end]][1], muscle_theta_r[muscle[2:end]][2], 
            text=uppercase(muscle[2:end]), color=muscle_colors[i][2],
            fontsize=22
        )
    end
end
hist!(ax_muscle_hist, phase_dict[max_mi_neuron[thismoth]] .* 2*pi, 
    bins=ceil(Int, sqrt(length(phase_dict[max_mi_neuron[thismoth]]))), 
    normalization=:pdf, offset=-1, scale_to=1.0, color=Makie.wong_colors()[1])
lines!(ax_muscle_hist, range(0, 2*pi, 1000), zeros(1000), color=:black, linewidth=2)
scatter!(ax_muscle_hist, 0, -1, color=:black, markersize=5)
text!(ax_muscle_hist, pi/2, -0.6, text="N. 76", font=:bold, color=:black, fontsize=18, align=(:center, :center))

lines!(ax_muscle_hist, [0, 0], [0.92, 0.98], color=:black, linewidth=3)
text!(ax_muscle_hist, 0, 1., text="LDLM", color=:black, fontsize=22, align=(:left, :center))

apply_letter_label(gb, "B")

# -------- Circular histograms of all "good" neurons in a moth
phase_dict, wblen_dict, muscle_phase_dict = get_phase_dict(thismoth)
gc = top_row[1:2, 2] = GridLayout()

# 1. Selection of good neurons (not MUA) from moth 1
phase_dict, wblen_dict, muscle_phase_dict = get_phase_dict(thismoth)
df_neuron = @pipe df |> 
    @subset(_, :moth .== thismoth, :label .== "good") |> 
    @subset(_, :peak_mi, :nspikes .> 1000, :muscle .== "all") |> 
    @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi))
sort!(df_neuron, [order(:label), order(:mi)])
# 2. For each neuron, draw polar histogram
polar_axes = []
polar_mis = Float64[]
for (i,row) in enumerate(eachrow(df_neuron))
    ri, ci = mod(i+1, 2) + 1, repeat(1:nrow(df_neuron), inner=2)[i]
    neur = string(round(Int, row.neuron))
    thisax = PolarAxis(top_row[1:2,2], 
        width=Relative(0.3), height=Relative(0.3), halign=-0.271 + 0.075*i, valign=0.075 + 0.85 * (ri-1),
        clip=false, clip_r=false
    )
    hiderdecorations!(thisax)
    hidethetadecorations!(thisax, grid=false, minorgrid=false)
    if neur in example_neurons
        hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[neur]))), 
            color=example_neuron_colors[neur])
        scatter!(thisax, 0, 0, color=:black, markersize=6)
    else
        hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[neur]))),
            color=:grey)
        scatter!(thisax, 0, 0, color=:black, markersize=6)
    end
    push!(polar_axes, thisax)
    push!(polar_mis, max(row.mi, 0.0))
end
# 3. Add the number line axis on the same grid cell
ax_nl = Axis(top_row[1:2,2], 
    width=Relative(1.0), height=Relative(1.0),
    halign=0.5, valign=0.5,
    backgroundcolor=:transparent
)
hidedecorations!(ax_nl)
hidespines!(ax_nl)
mi_min, mi_max = -0.1, maximum(polar_mis) * 1.1
y_min, y_max = 0.0, 1.0
xlims!(ax_nl, mi_min, mi_max)
ylims!(ax_nl, y_min, y_max)
line_pos = 0.5
lines!(ax_nl, [0, mi_max], [line_pos, line_pos], color=:black, linewidth=2) # Central number line
poly!(ax_nl, Point2f[(mi_max, line_pos),(mi_max-0.2, line_pos + 0.025), (mi_max-0.2, line_pos - 0.025)], color=:black)
for tick in [0.0, 10.0, 20.0]
    lines!(ax_nl, [tick, tick], line_pos .+ [-0.025, 0.025], color=:black, linewidth=2) # End tick
    textlabel!(ax_nl, tick, line_pos - 0.05, text=string(convert(Int, tick)), strokewidth=0)
end
text!(ax_nl, 0.7, 0.55, text="Mutual Information I(X;Y) (bits/s)", fontsize=22, align=(:left, :center), space=:relative)
# 4. Connect lines from polar hist plots to number line
on(events(f).window_open) do _  # fires once layout is resolved
    ax_nl_bb = ax_nl.layoutobservables.computedbbox[]
    ax_nl_origin = ax_nl_bb.origin
    ax_nl_widths = ax_nl_bb.widths
    for (pa, mi, i, neur) in zip(polar_axes, polar_mis, 1:length(polar_axes), string.(round.(Int, df_neuron.neuron)))
        pa_bb = pa.layoutobservables.computedbbox[]
        # Center of polar axis in scene pixels
        if mod(i,2) == 0 # Bottom row of polar axes
            px_center = pa_bb.origin .+ [pa_bb.widths[1], 0] ./ 2
        else
            px_center = pa_bb.origin .+ [pa_bb.widths[1] / 2, pa_bb.widths[2]]
        end
        # Convert scene pixels → ax_nl data coords
        x_frac = (px_center[1] - ax_nl_origin[1]) / ax_nl_widths[1]
        y_frac = (px_center[2] - ax_nl_origin[2]) / ax_nl_widths[2]
        pm = mod(i,2) == 0 ? 1 : -1
        x_data = mi_min + x_frac * (mi_max - mi_min)
        y_data = y_min + y_frac * y_max  + pm*0.01 # ylims are (0,1)
        # MI position on number line
        color = neur in example_neurons ? example_neuron_colors[neur] : :gray
        lines!(ax_nl, [x_data, mi], [y_data, line_pos], color=color, linewidth=0.8)
        scatter!(ax_nl, mi, line_pos, color=color, markersize=15)
        # Label position: centered on polar axis x, just above/below it
        px_x_center = pa_bb.origin[1] + pa_bb.widths[1] / 2
        x_label = mi_min + ((px_x_center - ax_nl_origin[1]) / ax_nl_widths[1]) * (mi_max - mi_min)
        # Label for neuron polar plot
        neur = string(round(Int, df_neuron[i, :neuron]))
        text_color = neur in example_neurons ? example_neuron_colors[neur] : :black
        if mod(i, 2) == 0  # top row: label above
            px_top = pa_bb.origin[2] + pa_bb.widths[2]
            y_frac = (px_top - ax_nl_origin[2]) / ax_nl_widths[2]
            y_label = y_min + y_frac * y_max
            text!(ax_nl, x_label, y_label, text="N. " * neur,
                align=(:center, :bottom), fontsize=20, color=text_color, font=:bold)
        else  # bottom row: label below
            px_bottom = pa_bb.origin[2]
            y_frac = (px_bottom - ax_nl_origin[2]) / ax_nl_widths[2]
            y_label = y_min + y_frac * y_max
            if neur == "82"
                x_label += 0.15
            end
            text!(ax_nl, x_label, y_label, text="N. " * neur,
                align=(:center, :top), fontsize=20, color=text_color, font=:bold)
        end
    end
end

apply_letter_label(gc, "C")
rowgap!(top_row, 0)
rowsize!(top_row, 1, Relative(1/4))
colsize!(top_row, 2, Relative(2/3))


bottom_row = f[2, 1:3] = GridLayout()
gd = bottom_row[1, 2] = GridLayout()
ge = bottom_row[1, 1] = GridLayout()
gf = bottom_row[1, 3] = GridLayout()


# -------- Power as predictor of MI, precision
wingbeat_freq_range = [1, 100]

spec_range = function(neur, spikes; wingbeat_freq_range=[1,200], fsamp=fsamp)
    discrete_vec = zeros(spikes[neur][end]+10)
    discrete_vec[spikes[neur]] .= 1
    pxx = welch_pgram(discrete_vec, fsamp * 10; fs=fsamp)

    fi = findfirst(pxx.freq .> wingbeat_freq_range[1])
    li = findlast(pxx.freq .< wingbeat_freq_range[2])
    return pxx.freq[fi:li], pxx.power[fi:li]
end

ax_freq = Axis(gd[1,1], xlabel="Frequency (Hz)", ylabel="Power spectral density (dB/Hz)")
vspan!(ax_freq, 17, 21, color=RGBf(0.8, 0.8, 0.8))
# Spectra, histogram for each neuron
power_vals = zeros(length(example_neurons))
for (i, neur) in enumerate(example_neurons)
    pfreq, ppower = spec_range(neur, spikes; wingbeat_freq_range=wingbeat_freq_range)
    mask = (pfreq .> 12) .&& (pfreq .< 24)
    ind = findmax(ppower[mask])[2] + findfirst(mask) - 1
    bump = neur .== "27" ? 1.0 : 0.0
    lines!(ax_freq, pfreq, 10 .* log10.(ppower) .+ bump, color=example_neuron_colors[neur])
    scatter!(ax_freq, pfreq[ind], 10 .* log10.(ppower[ind]) .+ bump, color=:black, markersize=15)
    scatter!(ax_freq, pfreq[ind], 10 .* log10.(ppower[ind]) .+ bump, color=example_neuron_colors[neur], markersize=10)
    power_vals[i] = 10 .* log10.(ppower[ind]) .+ bump
end
xlims!(ax_freq, 0, 50)

# Phase histograms ordered/sorted by peak power
ax_power_sort = Axis(gd[2,1])
hidedecorations!(ax_power_sort)
hidespines!(ax_power_sort)
power_polar_axes = []
for (i, examp_neuron) in enumerate(example_neurons[sortperm(power_vals)])
    this_inset = PolarAxis(gd[2,1], 
        width=Relative(circ_sort_rel_size), height=Relative(circ_sort_rel_size), halign=-0.5 + 0.4 * i, valign=0.5,
        clip=false, clip_r=false
    )
    hidedecorations!(this_inset)
    hist!(this_inset, phase_dict[examp_neuron] .* 2*pi, 
        normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[examp_neuron]))),
        color=example_neuron_colors[examp_neuron]
    )
    scatter!(this_inset, 0, 0, color=:black, markersize=5)
    push!(power_polar_axes, this_inset)
end
# Put "axis" line, labels for peak power
xmin, xmax = -81, -68
ymin, ymax = -0.25, 0.8
arrows2d!(ax_power_sort, Point2f(xmin, 0), [[xmax-xmin,0]])
ticks = Makie.get_tickvalues(LinearTicks(3), identity, xmin, xmax)
for lab_val in ticks
    lines!(ax_power_sort, [lab_val, lab_val], [-0.04, 0.04], color=:black)
    text!(ax_power_sort, lab_val, -0.06, text=string(lab_val), align=(:center, :top))
end
Label(gd[2,1,Bottom()], "Power at \n wingbeat frequency (dB/Hz)", fontsize=18, tellwidth=false, padding=(0,0,0,-20))
limits!(ax_power_sort, (xmin, xmax), (ymin, ymax))
let xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    on(events(f).window_open) do _  # fires once layout is resolved
        ax_p_bb = ax_power_sort.layoutobservables.computedbbox[]
        ax_p_origin = ax_p_bb.origin
        ax_p_widths = ax_p_bb.widths
        psort = sortperm(power_vals)
        for (pa, pval, neur) in zip(power_polar_axes, power_vals[psort], example_neurons[psort])
            pa_bb = pa.layoutobservables.computedbbox[]
            # Top of polar axis in scene pixels
            px_center = pa_bb.origin .+ [pa_bb.widths[1] / 2, pa_bb.widths[2]]
            # Convert scene pixels → ax_power_sort data coords
            x_data = xmin + ((px_center[1] - ax_p_origin[1]) / ax_p_widths[1]) * (xmax - xmin)
            y_data = ymin + ((px_center[2] - pa_bb.widths[2]/2 - ax_p_origin[2]) / ax_p_widths[2]) * ymax  # will require fine tuning
            # Power position on number line
            color = example_neuron_colors[neur]
            lines!(ax_power_sort, [x_data, pval], [y_data, 0], color=color, linewidth=0.8)
            sc = scatter!(ax_power_sort, pval, 0, color=color, markersize=15)
            translate!(sc, 0, 0, 10)
            # Label position: centered on polar axis x, just above/below it
            x_label = xmin + ((px_center[1] - ax_p_origin[1]) / ax_p_widths[1]) * (xmax - xmin)
            # Label above neuron polar plot
            px_top = pa_bb.origin[2] + pa_bb.widths[2]
            y_label = ymin + ((px_top - ax_p_origin[2]) / ax_p_widths[2]) * ymax + 0.15
            text!(ax_power_sort, x_label, y_label, text="N. " * neur,
                align=(:center, :bottom), fontsize=18, color=color, font=:bold)
        end
    end
end

df_power = @pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all", :nspikes .> 0) |> 
leftjoin(_, dfc, on=[:moth, :neuron])

gd_right = gd[1:2, 2] = GridLayout()
ax_power_mi = Axis(gd_right[1,1], yscale=log10, ylabel="I(X;Y) (bits/s)")
ax_power_prec = Axis(gd_right[2,1], yscale=log10, xlabel="Power at wingbeat frequency (dB/Hz)", ylabel="Spike timing precision (ms)")

for timing_info in ["Spike timing info", "No spike timing info"]
    gdf = @subset(df_power, :has_timing_info .== timing_info)
    scatter!(ax_power_mi, 10 .* log10.(gdf.peak_power), gdf.mi, 
        color=timing_colors[gdf.has_timing_info[1]],
        label=gdf.has_timing_info[1]
        # label=titlecase(first(gdf.direction)), 
        # color=color_dict[first(gdf.direction)]
    )
    if timing_info .== "Spike timing info"
        # mask = gdf.has_timing_info .== "Spike timing info"
        scatter!(ax_power_prec, 10 .* log10.(gdf.peak_power), gdf.precision,
            color=timing_colors[gdf.has_timing_info[1]])
            # color=color_dict[first(gdf.direction)])
    end
end
axislegend(ax_power_mi, position=:rb, backgroundcolor=(:white, 1.0))
hidexdecorations!(ax_power_mi, grid=false, minorgrid=false)

apply_letter_label(gd, "Ei")
Label(gd[2,1,TopLeft()], "ii",
    fontsize = 26,
    font = :bold,
    padding = (0, 0, 0, 0),
    halign = :right,
    tellwidth = false,
    tellheight = false
)
apply_letter_label(gd_right, "iii")

linkxaxes!(ax_power_mi, ax_power_prec)
rowgap!(gd_right, 0)
colsize!(gd, 2, Relative(0.6))

# -------- Circularity as predictor of MI, precision

df_circ = @pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all", :nspikes .> 1000) |> 
leftjoin(_, dfc, on=[:moth, :neuron]) |> 
@subset(_, :omnibus_stat .!= 0)

ecdf_func = function(phase_spikes, x)
    sorted = sort(phase_spikes .* 2 * pi)
    return searchsortedlast.(Ref(sorted), x) ./ length(sorted)
end

ax_ecdf_examp = Axis(ge[:,1], 
    xlabel="Phase angle (Radians)", ylabel="Cumulative probability",
    xticks=([0, pi/2, pi, 3*pi/2, 2*pi], ["0", "π/2", "π", "3π/2", "2π"]),
    limits=((0,2*pi), (0,1))
)
ge_right = ge[1:2, 2] = GridLayout()
ax_circ_mi = Axis(ge_right[1,1], xscale=log10, yscale=log10, ylabel="I(X;Y) (bits/s)")
ax_circ_prec = Axis(ge_right[2,1], xscale=log10, yscale=log10, xlabel="Kuiper statistic V", ylabel="Spike timing precision (ms)")

unif_vec = range(0, 1, 1000)
for (i, examp_neuron) in enumerate(example_neurons[[1,3,4,2]]) # run in different order to get example N. 27 on top
    xvec = range(0, 2*pi, 1000)
    ecdf = ecdf_func(phase_dict[examp_neuron], xvec)
    comparison = ecdf .- unif_vec
    maxind = findmax(comparison)[2]
    minind = findmin(comparison)[2]
    lines!(ax_ecdf_examp, repeat([unif_vec[maxind] * 2*pi], 2), [unif_vec[maxind], ecdf[maxind]], 
        color=example_neuron_colors[examp_neuron], linewidth=1.5,
    )
    lines!(ax_ecdf_examp, repeat([unif_vec[minind] * 2*pi], 2), [unif_vec[minind], ecdf[minind]], 
        color=example_neuron_colors[examp_neuron], linewidth=1.5,
    )
    lines!(ax_ecdf_examp, xvec, ecdf, color=example_neuron_colors[examp_neuron], linewidth=2)
end
text!(ax_ecdf_examp, pi*0.97, 0.32, text=L"D^-", color=example_neuron_colors["27"], fontsize=19)
text!(ax_ecdf_examp, 7*pi/4-pi/8-0.1, 0.895, text=L"D^+", color=example_neuron_colors["27"], fontsize=19)
textlabel!(ax_ecdf_examp, pi/2+0.5, 0.85, text=L"V=D^+ + D^-", fontsize=20)

# Phase histograms ordered/sorted by circularity
ax_circ_sort = Axis(ge[2,1])
hidedecorations!(ax_circ_sort)
hidespines!(ax_circ_sort)
examp_circ = @subset(df_circ, :moth .== thismoth, string.(round.(Int, :neuron)) .∈ Ref(example_neurons))
examp_circ.neuron = string.(round.(Int, examp_circ.neuron))
sort!(examp_circ, [order(:kuiper_stat)])
circ_polar_axes = []
for (i, examp_neuron) in enumerate(examp_circ.neuron)
    this_inset = PolarAxis(ge[2,1], 
        width=Relative(circ_sort_rel_size), height=Relative(circ_sort_rel_size), halign=-0.5 + 0.4 * i, valign=0.5,
        clip=false, clip_r=false
    )
    hidedecorations!(this_inset)
    hist!(this_inset, phase_dict[examp_neuron] .* 2*pi, 
        normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[examp_neuron]))),
        color=example_neuron_colors[examp_neuron]
    )
    scatter!(this_inset, 0, 0, color=:black, markersize=5)
    push!(circ_polar_axes, this_inset)
end
# Put "axis" line, labels for Kuiper stat
xmin, xmax = 5, 40
ymin, ymax = -0.25, 0.8
arrows2d!(ax_circ_sort, Point2f(xmin, 0), [[xmax-xmin,0]])
ticks = Makie.get_tickvalues(LinearTicks(3), identity, xmin, xmax)
for lab_val in ticks
    lines!(ax_circ_sort, [lab_val, lab_val], [-0.04, 0.04], color=:black)
    text!(ax_circ_sort, lab_val, -0.06, text=string(lab_val), align=(:center, :top))
end
# text!(ax_circ_sort, 0.5, 0.05, text="Kuiper statistic V", space=:relative, align=(:center, :center))
Label(ge[2,1,Bottom()], "Kuiper statistic V", fontsize=18, tellwidth=false, padding=(0,0,0,-35))
limits!(ax_circ_sort, (xmin, xmax), (ymin, ymax))
let xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    on(events(f).window_open) do _  # fires once layout is resolved
        ax_p_bb = ax_circ_sort.layoutobservables.computedbbox[]
        ax_p_origin = ax_p_bb.origin
        ax_p_widths = ax_p_bb.widths
        for (pa, pval, neur) in zip(circ_polar_axes, examp_circ.kuiper_stat, examp_circ.neuron)
            pa_bb = pa.layoutobservables.computedbbox[]
            # Top of polar axis in scene pixels
            px_center = pa_bb.origin .+ [pa_bb.widths[1] / 2, pa_bb.widths[2]]
            # Convert scene pixels → ax_circ_sort data coords
            x_data = xmin + ((px_center[1] - ax_p_origin[1]) / ax_p_widths[1]) * (xmax - xmin)
            y_data = ymin + ((px_center[2] - pa_bb.widths[2]/2 - ax_p_origin[2]) / ax_p_widths[2]) * ymax  # will require fine tuning
            # Power position on number line
            color = example_neuron_colors[neur]
            lines!(ax_circ_sort, [x_data, pval], [y_data, 0], color=color, linewidth=0.8)
            sc = scatter!(ax_circ_sort, pval, 0, color=color, markersize=15)
            translate!(sc, 0, 0, 10)
            # Label position: centered on polar axis x, just above/below it
            x_label = xmin + ((px_center[1] - ax_p_origin[1]) / ax_p_widths[1]) * (xmax - xmin)
            # Label above neuron polar plot
            px_top = pa_bb.origin[2] + pa_bb.widths[2]
            y_label = ymin + ((px_top - ax_p_origin[2]) / ax_p_widths[2]) * ymax + 0.15
            text!(ax_circ_sort, x_label, y_label, text="N. " * neur,
                align=(:center, :bottom), fontsize=18, color=color, font=:bold)
        end
    end
end
lines!(ax_ecdf_examp, [0, 2*pi], [0, 1], color=:grey, linewidth=3.2) # Uniform distribution

for timing_info in ["Spike timing info", "No spike timing info"]
    gdf = @subset(df_circ, :has_timing_info .== timing_info)
    scatter!(ax_circ_mi, gdf.kuiper_stat, gdf.mi, 
        label=gdf.has_timing_info[1],
        color=timing_colors[gdf.has_timing_info[1]]
        # label=titlecase(direction), 
        # color=color_dict[direction]
    )
    if timing_info .== "Spike timing info"
        # mask = gdf.has_timing_info .== "Spike timing info"
        scatter!(ax_circ_prec, gdf.kuiper_stat, gdf.precision,
            color=timing_colors[gdf.has_timing_info[1]]
            # color=color_dict[direction]
        )
    end
end
# axislegend(ax_circ_mi, position=:rb, backgroundcolor=(:white, 0.5))
hidexdecorations!(ax_circ_mi, grid=false, minorgrid=false)

apply_letter_label(ge, "Di")
Label(ge[2,1,TopLeft()], "ii",
    fontsize = 26,
    font = :bold,
    padding = (0, 0, 0, 0),
    halign = :right,
    tellwidth = false,
    tellheight = false
)
apply_letter_label(ge_right, "iii")

linkxaxes!(ax_circ_mi, ax_circ_prec)
rowgap!(ge_right, 0)
colsize!(ge, 2, Relative(0.6))

# -------- Final panel showing how two methods disagree on neurons without timing information
dt = @pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all", :nspikes .> 1000) |> 
leftjoin(_, dfc, on=[:moth, :neuron])

ax_stat_comp = Axis(gf[1,1], 
    xlabel="Kuiper statistic V", ylabel="Power at wingbeat frequency (dB/Hz)",
    xscale=log10,
    aspect=1
)
for gdf in groupby(dt, :has_timing_info)
    scatter!(ax_stat_comp, gdf.kuiper_stat, 10 .* log10.(gdf.peak_power), 
        label=gdf.has_timing_info[1], color=timing_colors[gdf.has_timing_info[1]])
end
axislegend(ax_stat_comp, position=:lt)
xlims!(ax_stat_comp, 0.86, 70)
apply_letter_label(gf, "F")

# Final adjustments 
colgap!(bottom_row, 20)
colsize!(bottom_row, 3, Relative(1/5))
rowsize!(f.layout, 2, Relative(0.5))
return f
end

fontsize_theme = Theme(fontsize = 18)
f = with_theme(fontsize_theme) do
    phasic_neuron_figure()
end
display(f)
save(joinpath(fig_dir, "fig_phasic.png"), f)


## Supplement Figure: Other circularity tests
CairoMakie.activate!()

timing_colors = Dict("No spike timing info" => "#5e3c99", "Spike timing info" => "#e66101")

df_circ = @pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all", :nspikes .> 1000) |> 
leftjoin(_, dfc, on=[:moth, :neuron]) |> 
@subset(_, :omnibus_stat .!= 0)


tests = ["kuiper", "watson", "rao", "omnibus"]

f = Figure(size=(1000,400))
ax_mi = [Axis(f[1,i], xscale=log10, yscale=log10) for i in 1:length(tests)]
ax_prec = [Axis(f[2,i], xscale=log10, yscale=log10) for i in 1:length(tests)]

for (i, test) in enumerate(tests)
    for timing_info in ["Spike timing info", "No spike timing info"]
        gdf = @subset(df_circ, :has_timing_info .== timing_info)
        scatter!(ax_mi[i], gdf[:, test * "_stat"], gdf.mi,
            label=gdf.has_timing_info[1],
            color=timing_colors[gdf.has_timing_info[1]]
        )
        if timing_info == "Spike timing info"
            scatter!(ax_prec[i], gdf[:, test * "_stat"], gdf.precision,
                label=gdf.has_timing_info[1],
                color=timing_colors[gdf.has_timing_info[1]]
            )
        end
    end
    linkxaxes!(ax_mi[i], ax_prec[i])
    hidexdecorations!(ax_mi[i], grid=false)
    ax_prec[i].xlabel = titlecase(test) * " test statistic"
end
linkyaxes!(ax_prec...)
linkyaxes!(ax_mi...)

ax_mi[1].ylabel = "I(X;Y) (bits/s)"
ax_prec[1].ylabel = "τ(X;Y) (ms)"

f[0,:] = Legend(f, ax_mi[end], grid=false, minorgrid=false, orientation = :horizontal)

display(f)

save(joinpath(fig_dir, "fig_supp_circularity_tests.pdf"), f)

##
@pipe df_circ |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:kuiper_stat, :mi, color=:direction) * visual(Scatter)
) |> 
draw(_, axis=(; xscale=log10, yscale=log10))

##


dfc = DataFrame(Arrow.Table(joinpath(data_dir, "..", "circular_stats.arrow")))
disallowmissing!(dfc)
vector_cols = ["movm_mu", "kappa", "movm_rvec", "movm_BIC", "mokj_mu", "gamma", "rho", "lam", "mokj_BIC"]
# Clean up types
for col in vector_cols
    dfc[!,col] = [convert(Vector{Float64}, v) for v in dfc[!,col]]
end
transform!(dfc, :moth => ByRow(x -> replace(x, r"_1$" => "-1")) => :moth)

α = 0.05
moth_pval_threshold = Dict()
# Example plot for omnibus test
f = Figure(size=(300,700))
for (i,dt) in enumerate(groupby(@subset(dfc, :omnibus_p .!= 0), :moth))
    ax = Axis(f[i,1], title=dt.moth[1])
    p = sort(dt.omnibus_p)
    mask = p .<= (collect(1:nrow(dt)) / nrow(dt) * α)
    nonzero = findall(p .!= 0)
    lastind = findlast(p[nonzero] .<= (collect(1:length(p[nonzero])) / length(p[nonzero]) * α))
    lastind = min(length(p[nonzero])-1, lastind)
    moth_pval_threshold[dt.moth[1]] = p[lastind] + (p[lastind+1] - p[lastind])/2
    scatterlines!(ax, collect(1:nrow(dt))[mask], p[mask], color=Makie.wong_colors()[1])
    scatterlines!(ax, collect(1:nrow(dt))[(!).(mask)], p[(!).(mask)], color=:red)
end
display(f)

## Linear models between metrics (Kuiper, PSD@wbf) and mutual info/precision

df_circ = @pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all", :nspikes .> 1000) |> 
leftjoin(_, dfc, on=[:moth, :neuron]) |> 
@subset(_, :omnibus_stat .!= 0)

models = [
    lm(@formula(log(mi) ~ log(kuiper_stat) * has_timing_info), df_circ)
    lm(@formula(log(mi) ~ log(kuiper_stat) + has_timing_info), df_circ)
    lm(@formula(log(mi) ~ log(kuiper_stat)), df_circ)
    lm(@formula(log(mi) ~ 1), df_circ)
]
println(bic.([mod.model for mod in models]))
println(ftest([mod.model for mod in models]...))

df_circ_prec = @subset(df_circ, :has_timing_info .== "Spike timing info")
models = [
    lm(@formula(log(precision) ~ log(kuiper_stat)), df_circ_prec)
    lm(@formula(log(precision) ~ 1), df_circ_prec)
]
println(bic.([mod.model for mod in models]))
println(ftest([mod.model for mod in models]...))


models = [
    lm(@formula(log(mi) ~ log(kuiper_stat) * has_timing_info), df_circ)
    lm(@formula(log(mi) ~ log(kuiper_stat) + has_timing_info), df_circ)
    lm(@formula(log(mi) ~ log(kuiper_stat)), df_circ)
    lm(@formula(log(mi) ~ 1), df_circ)
]

# Power spectral density
df_circ = @transform(df_circ, :psd = 10 .* log10.(:peak_power))
models = [
    lm(@formula(log(mi) ~ psd * has_timing_info), df_circ)
    lm(@formula(log(mi) ~ psd + has_timing_info), df_circ)
    lm(@formula(log(mi) ~ psd), df_circ)
    lm(@formula(log(mi) ~ 1), df_circ)
]
println(ftest([mod.model for mod in models]...))

df_circ_prec = @subset(df_circ, :has_timing_info .== "Spike timing info")
models = [
    lm(@formula(log(precision) ~ psd), df_circ_prec)
    lm(@formula(log(precision) ~ 1), df_circ_prec)
]
println(ftest([mod.model for mod in models]...))

# Compare two metrics


##

@pipe df |> 
@groupby(_, [:moth, :neuron, :muscle]) |> 
@transform(_, :has_timing_info = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :mi .> 0, :peak_mi, :muscle .== "all", :nspikes .> 1000) |> 
(
AlgebraOfGraphics.data(_) * mapping(:meanrate, :precision, color=:moth) * visual(Scatter)
) |> 
draw(_)