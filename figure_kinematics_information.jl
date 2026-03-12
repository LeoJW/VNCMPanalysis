using Logging

"""
Kinematics figure


This script requires running the first section of main_figures.jl and main_spike_figures.jl to get the necessary imports, dataframes, 
and other things set up and loaded. It's just put here to keep main_figures.jl from becoming 10,000 lines long
"""

CairoMakie.activate!()

function figure_kinematics_info_alt()
inch = 96
cm = inch / 2.54
f = Figure(size=(8.5cm, 15cm))


dt = @pipe df |> 
@subset(_, :nspikes .> 1000, :peak_mi, :moth .∈ Ref(["2025-02-25", "2025-02-25-1"])) |> 
leftjoin(_, @subset(df_kine, :peak_mi), on=[:muscle, :moth], renamecols=""=>"_YZ") |> 
leftjoin(_, @subset(df_kine_neur, :peak_mi), on=[:neuron, :moth], renamecols=""=>"_XZ") |> 
@transform(_, :mi_XY = :mi) |> 
@transform(_, :n2m_loss = :mi_XY .- :mi_XZ) |> 
@transform(_, :m2k_loss = :mi_YZ .- :mi_XZ) |> 
@subset(_, (!).(ismissing.(:mi_XY)), (!).(ismissing.(:mi_XZ))) |> 
@subset(_, (!).(isnan.(:mi_XY)) .&& (!).(isnan.(:mi_XZ)))
dt.bilat_muscle = [x[2:end] for x in dt.muscle]
dt = @pipe dt |> 
@transform!(_, :muscle = ifelse.(:single, :bilat_muscle, :muscle)) |> 
@subset(_, (!).(:muscle .∈ Ref(["Rsteering", "Rpower", "Lsteering", "Lpower"]))) |> 
@subset(_, abs.(:precision_XZ .- :precision) .< 50)

dtp = @pipe df |> 
@subset(_, :nspikes .> 1000, :peak_valid_mi, :moth .∈ Ref(["2025-02-25", "2025-02-25-1"])) |> 
leftjoin(_, @subset(df_kine, :peak_valid_mi), on=[:muscle, :moth], renamecols=""=>"_YZ") |> 
leftjoin(_, @subset(df_kine_neur, :peak_valid_mi), on=[:neuron, :moth], renamecols=""=>"_XZ") |> 
@transform(_, :mi_XY = :mi) |> 
@transform(_, :n2m_loss = :mi_XY .- :mi_XZ) |> 
@transform(_, :m2k_loss = :mi_YZ .- :mi_XZ) |> 
@subset(_, (!).(ismissing.(:mi_XY)), (!).(ismissing.(:mi_XZ))) |> 
@subset(_, (!).(isnan.(:mi_XY)) .&& (!).(isnan.(:mi_XZ)))
dtp.bilat_muscle = [x[2:end] for x in dtp.muscle]
dtp = @pipe dtp |> 
@transform!(_, :muscle = ifelse.(:single, :bilat_muscle, :muscle)) |> 
@subset(_, (!).(:muscle .∈ Ref(["Rsteering", "Rpower", "Lsteering", "Lpower"]))) |> 
@subset(_, abs.(:precision_XZ .- :precision) .< 50)

# Common elements for raincloud plots
sort_dict = Dict(
    "ax"=>"a", "ba"=>"b", "sa"=>"c", "dvm"=>"d", "dlm"=>"e", "steering"=>"f", "power"=>"g", "all"=>"h"
)
# Colors are in reverse order due to how I use indexin() later
colors = [
    "#707070", "#9e9e9e", "#9e9e9e",
    muscle_colors_dict["ldlm"], muscle_colors_dict["ldvm"],
    muscle_colors_dict["lsa"], muscle_colors_dict["lba"], muscle_colors_dict["lax"], 
]

ga = f[1,1] = GridLayout()
gb = f[1,2] = GridLayout()
colgap!(f.layout, 0)

ax = Axis(ga[1,1], xlabel="I(X;Z) - I(X;Y) (bits/s)")
hideydecorations!(ax)
hidespines!(ax, :l, :r, :t)

jitter_width = 0.1

# Rainclouds for mutual information
# Rainclouds sorts using unique(), so create category vector that orders how I want
category = getindex.(Ref(sort_dict), dt.muscle)
inds = sortperm(category, rev=true)
vals = dt.mi_XZ .- dt.mi_XY
# rainclouds!(ax, category[inds], vals[inds],
#     orientation=:horizontal, plot_boxplots=false,
#     clouds=hist, hist_bins=40,
#     cloud_width = 1., jitter_width=0.1,
#     color=colors[indexin(category[inds], unique(category[inds]))],
# )
# Raincloud plots + mean lines on histograms for MI
for (i,cat) in enumerate(unique(category[inds]))
    mask = category[inds] .== cat
    # Plot scatter, hist
    hist_bins = range(-12, 12, 31) # odd num. of bins gives clean separation at zero
    hist!(ax, vals[inds][mask], bins=hist_bins, 
        offset=i, scale_to=0.8,
        color=colors[i])
    scatter!(ax, vals[inds][mask], i .- rand(sum(mask)) .* jitter_width .- 0.02,
        markersize=2.0, color=colors[i])
    
    # Calculate histogram heights, add mean line
    mean_val = mean(vals[inds][mask])
    thishist = fit(Histogram, vals[inds][mask], hist_bins)
    heights = thishist.weights ./ maximum(thishist.weights) .* 0.8
    bin_centers = (hist_bins[1:end-1] .+ hist_bins[2:end]) ./ 2
    mean_bin_idx = findmin(abs.(bin_centers .- mean_val))[2]
    mean_line_height = heights[mean_bin_idx]
    lines!(ax, [mean_val, mean_val], [i, i+mean_line_height], color=:black)
end

ax2 = Axis(gb[1,1], xlabel="τ(X;Y) - τ(X;Z) (ms)", xticks=[-10, -5, 0, 5, 10])
hideydecorations!(ax2)
hidespines!(ax2, :l, :r, :t)

# Rainclouds for precision
category = getindex.(Ref(sort_dict), dtp.muscle)
inds = sortperm(category, rev=true)
vals = dtp.precision .- dtp.precision_XZ
# rainclouds!(ax2, category[inds], vals[inds],
#     orientation=:horizontal, plot_boxplots=false,
#     clouds=hist, hist_bins=40,
#     cloud_width = 1., jitter_width=0.1,
#     color=colors[indexin(category[inds], unique(category[inds]))],
# )
# Raincloud plots + mean lines on histograms for precision
for (i,cat) in enumerate(unique(category[inds]))
    mask = category[inds] .== cat
    # Plot scatter, hist
    hist_bins = range(-12, 12, 31) # odd num. of bins gives clean separation at zero
    hist!(ax2, vals[inds][mask], bins=hist_bins, 
        offset=i, scale_to=0.8,
        color=colors[i])
    scatter!(ax2, vals[inds][mask], i .- rand(sum(mask)) .* jitter_width .- 0.02,
        markersize=2.0, color=colors[i])
    
    # Calculate histogram heights, add mean line
    mean_val = mean(vals[inds][mask])
    thishist = fit(Histogram, vals[inds][mask], hist_bins)
    heights = thishist.weights ./ maximum(thishist.weights) .* 0.8
    bin_centers = (hist_bins[1:end-1] .+ hist_bins[2:end]) ./ 2
    mean_bin_idx = findmin(abs.(bin_centers .- mean_val))[2]
    mean_line_height = heights[mean_bin_idx]
    lines!(ax2, [mean_val, mean_val], [i, i+mean_line_height], color=:black)
end


# Muscle labels on left side
ylims = (0.8, 9.45)
ylims!(ax, ylims)
ylims!(ax2, ylims)
for (i,lab) in enumerate(["All", "Power", "Steering", "DLM", "DVM", "SA", "BA", "AX"])
    # Convert data coordinate i → fraction of axis height
    rel_y = (i - ylims[1]) / (ylims[2]-0.35 - ylims[1])
    fontcol = lab in ["DLM", "DVM", "SA", "BA", "AX"] ? muscle_colors_dict["r" * lowercase(lab)] : :black
    Label(ga[1,1], lab,
        tellheight = false, tellwidth = false,
        halign = :right, valign = rel_y + 0.015,
        justification=:right,
        padding = (0, 90, 0, 0),
        font = lab == "All" ? :bold : :regular,
        color=fontcol
    )
end

arr_height = 8.9
arrows2d!(ax, Point2f(0, arr_height), [[-7,0]], minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5)
arrows2d!(ax, Point2f(0, arr_height), [[+7,0]], minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5)
Label(ga[1,1], "More info.\nto muscle(s)",
    tellheight = false, tellwidth = false,
    halign = :left, valign = :top, justification=:right,
    padding = (-8, 0, 0, 0), # (l,r,b,t)
    fontsize=10
)
Label(ga[1,1], "More info.\nto wings",
    tellheight = false, tellwidth = false,
    halign = :right, valign = :top, justification=:left,
    padding = (0, 0, 0, 0), # (l,r,b,t)
    fontsize=10
)

arrows2d!(ax2, Point2f(0, arr_height), [[-7,0]], minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5)
arrows2d!(ax2, Point2f(0, arr_height), [[+7,0]], minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5)
Label(gb[1,1], "More precise\nto muscle(s)",
    tellheight = false, tellwidth = false,
    halign = :left, valign = :top, justification=:right,
    padding = (-10, 0, 0, 0), # (l,r,b,t)
    fontsize=10
)
Label(gb[1,1], "More precise\nto wings",
    tellheight = false, tellwidth = false,
    halign = :right, valign = :top, justification=:left,
    padding = (0, -9, 0, 0), # (l,r,b,t)
    fontsize=10
)

# Inset axes
for (i,lab) in enumerate(["all", "power", "steering", "DLM", "DVM", "SA", "BA", "AX"])
    ax_inset = Axis(f[:,:], 
        width=Relative(0.17), height=Relative(0.17), 
        halign=0.5, valign=0.14 * i - 0.18, 
        aspect=DataAspect()
    )
    img = with_logger(ConsoleLogger(stderr, Logging.Error)) do # suppress warnings about eXIf format
        load(assetpath(joinpath(fig_dir, "muscles_$(lab).png")))
    end
    image!(ax_inset, rotr90(img))
    hidedecorations!(ax_inset)
    hidespines!(ax_inset)
end

vlines!(ax, [0], color=:black)
vlines!(ax2, [0], color=:black)
xlims!(ax, -10, 10)
xlims!(ax2, -12, 12)

apply_letter_label(ga, "A"; fontsize=20)
apply_letter_label(gb, "B"; fontsize=20)

colsize!(f.layout, 1, Relative(0.5))
rowgap!(f.layout, 0)

f
end

f = figure_kinematics_info_alt()
display(f)

save(joinpath(fig_dir, "fig_kinematics_info.pdf"), f)