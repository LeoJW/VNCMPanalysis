using Logging

"""
Kinematics figure


This script requires running the first section of main_figures.jl and main_spike_figures.jl to get the necessary imports, dataframes, 
and other things set up and loaded. It's just put here to keep main_figures.jl from becoming 10,000 lines long
"""

CairoMakie.activate!()
# GLMakie.activate!()


function figure_kinematics_info()
inch = 96
cm = inch / 2.54
f = Figure(size=(17.8cm, 9cm))

color_dict = Dict("descending" => Makie.wong_colors()[1], "ascending" => Makie.wong_colors()[2], "uncertain" => Makie.wong_colors()[4])

dkn = @pipe df_kine_neur |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi)
# @subset(_, :peak_valid_mi)

dt = @pipe df |> 
@subset(_, :nspikes .> 1000, :peak_mi, :moth .∈ Ref(["2025-02-25", "2025-02-25-1"])) |> 
leftjoin(_, @subset(df_kine, :peak_mi), on=[:muscle, :moth], renamecols=""=>"_YZ") |> 
# @subset(_, :nspikes .> 1000, :peak_valid_mi, :moth .∈ Ref(["2025-02-25", "2025-02-25-1"])) |> 
# leftjoin(_, @subset(df_kine, :peak_valid_mi), on=[:muscle, :moth], renamecols=""=>"_YZ") |> 
leftjoin(_, dkn, on=[:neuron, :moth], renamecols=""=>"_XZ") |> 
@transform(_, :mi_XY = :mi) |> 
@transform(_, :n2m_loss = :mi_XY .- :mi_XZ) |> 
@transform(_, :m2k_loss = :mi_YZ .- :mi_XZ) |> 
@subset(_, (!).(ismissing.(:mi_XY)), (!).(ismissing.(:mi_XZ)))
# stack(_, [:mi_XY, :mi_XZ, :mi_YZ])

ga = f[1,1] = GridLayout()
gb = f[1,2] = GridLayout()
# Add a shaded box around DVM example
box_ad = Box(
    f[1, 1:2, Makie.GridLayoutBase.Outer()],
    alignmode = Outside(-10, -15, -12, -10),
    cornerradius = 8,
    color = (:lightblue, 0.15),
    strokecolor = (:steelblue, 0.8),
    strokewidth = 2,
)
# Move the box to the background so it doesn't cover the plots
Makie.translate!(box_ad.blockscene, 0, 0, -200)

ax_scatter = Axis(f[1,1], xlabel="I(X;DVM) (bits/s)", ylabel="I(X;Z) (bits/s)")
mask = [x[2:end] .== "dvm" for x in dt.muscle] .&& ((!).(isnan.(dt.mi_XY))) .&& ((!).(isnan.(dt.mi_XZ)))
ablines!(ax_scatter, 0, 1, color=:black, linewidth=2)
positions = [((dt[i,:mi_XY], dt[i,:mi_XZ]), (dt[i,:mi_XY], dt[i,:mi_XY])) for i in findall(mask)]
linesegments!(ax_scatter, positions, color=:gray)
scatter!(ax_scatter, dt[mask,:mi_XY], dt[mask,:mi_XZ], markersize=7)

# Inset axis
ax_inset = Axis(f[1,2], 
    width=Relative(0.5), height=Relative(0.5), 
    halign=0.9, valign=0.9, aspect=DataAspect())
img = with_logger(ConsoleLogger(stderr, Logging.Error)) do # suppress warnings about eXIf format
    load(assetpath(joinpath(fig_dir, "muscles_DVM.png")))
end
image!(ax_inset, rotr90(img))
hidedecorations!(ax_inset)
hidespines!(ax_inset)

# DVM hist
ax_hist = Axis(f[1,2], xlabel="I(X;Z) - I(X;DVM) (bits/s)", ylabel="Prob. density")
hist!(ax_hist, dt[mask,:mi_XZ] .- dt[mask,:mi_XY], normalization=:pdf)
vlines!(ax_hist, 0, color=:black)

xlims!(ax_scatter, low=0)
ylims!(ax_scatter, low=0)
ylims!(ax_hist, low=0)

# DLM hist
# gc = f[1,3]# = GridLayout()
ax_dlm_hist = Axis(f[1,3], xlabel="I(X;Z) - I(X;DLM) (bits/s)")
mask = [x[2:end] .== "dlm" for x in dt.muscle] .&& ((!).(isnan.(dt.mi_XY))) .&& ((!).(isnan.(dt.mi_XZ)))
hist!(ax_dlm_hist, dt[mask,:mi_XZ] .- dt[mask,:mi_XY], normalization=:pdf)
vlines!(ax_dlm_hist, 0, color=:black)
ylims!(ax_dlm_hist, low=0)
# DLM uCT inset
ax_dlm_inset = Axis(f[1,3], 
    width=Relative(0.5), height=Relative(0.5), 
    halign=0.9, valign=0.9, aspect=DataAspect())
img = with_logger(ConsoleLogger(stderr, Logging.Error)) do # suppress warnings about eXIf format
    load(assetpath(joinpath(fig_dir, "muscles_DLM.png")))
end
image!(ax_dlm_inset, rotr90(img))
hidedecorations!(ax_dlm_inset)
hidespines!(ax_dlm_inset)


# gd = f[2,1:3] = GridLayout()

# AX, BA, SA, DVM, DLM panels
ax_muscle = []
for (i,muscle) in enumerate(["ax", "ba", "sa"])
    if muscle .== "ax"
        thisax = Axis(f[2,i], 
            xlabel="I(X;Z) - I(X;" * uppercase(muscle) * ") (bits/s)", ylabel="Prob. density"
            # limits=(nothing, (-0.15, 0.5))
        )
    else
        thisax = Axis(f[2,i], xlabel="I(X;Z) - I(X;" * uppercase(muscle) * ") (bits/s)",
            # limits=(nothing, (-0.15, 0.5))
        )
        hideydecorations!(thisax, ticks=false, grid=false, minorgrid=false, minorticks=false)
    end
    push!(ax_muscle, thisax)
    mask = [x[2:end] .== muscle for x in dt.muscle] .&& ((!).(isnan.(dt.mi_XY))) .&& ((!).(isnan.(dt.mi_XZ)))
    hist!(thisax, dt[mask,:mi_XZ] .- dt[mask,:mi_XY], normalization=:pdf)
    # vals = dt[mask,:precision_XZ] .- dt[mask,:precision]
    # hist!(thisax, vals[abs.(vals) .< 50], normalization=:pdf)
    vlines!(thisax, [0], color=:black)

    # Inset axis
    ax_inset = Axis(f[2,i], 
        width=Relative(0.5), height=Relative(0.5), 
        halign=0.9, valign=0.9, aspect=DataAspect())
    img = with_logger(ConsoleLogger(stderr, Logging.Error)) do # suppress warnings about eXIf format
        load(assetpath(joinpath(fig_dir, "muscles_$(uppercase(muscle)).png")))
    end
    image!(ax_inset, rotr90(img))
    hidedecorations!(ax_inset)
    hidespines!(ax_inset)
end
linkaxes!(ax_muscle...)
[ylims!(ax, low=0) for ax in ax_muscle]

# All muscles panels
g_bottom = f[:,4] = GridLayout()
ge = g_bottom[1,1] = GridLayout()
gf = g_bottom[2,1] = GridLayout()

ax_all = Axis(ge[1,1], xlabel="I(X;Z) - I(X;Y) (bits/s)", ylabel="Prob. density", title="All muscles")
mask = dt.muscle .== "all"
hist!(ax_all, dt[mask, :mi_XZ] .- dt[mask, :mi_XY], normalization=:pdf)
vlines!(ax_all, [0], color=:black)
ylims!(ax_all, low=0)

# lastax = Axis(gc[1,1])
ax_prec_all = Axis(gf[1,1], xlabel="τ(X;Z) - τ(X;Y) (ms)")
ddt = @pipe df |> 
    @subset(_, :nspikes .> 1000, :peak_valid_mi, :moth .∈ Ref(["2025-02-25", "2025-02-25-1"])) |> 
    leftjoin(_, @subset(df_kine, :peak_valid_mi), on=[:muscle, :moth], renamecols=""=>"_YZ") |> 
    leftjoin(_, dkn, on=[:neuron, :moth], renamecols=""=>"_XZ") |> 
    @subset(_, :muscle .== "all", (!).(isnan.(:precision)), (!).(isnan.(:precision_XZ)))
hist!(ax_prec_all, ddt.precision_XZ .- ddt.precision, normalization=:pdf)
vlines!(ax_prec_all, [0], color=:black)
ylims!(ax_prec_all, low=0)

linkaxes!(ax_hist, ax_dlm_hist, ax_all, ax_muscle...)

# apply_letter_label(ga, "A")
# apply_letter_label(gb, "B")
# apply_letter_label(gc, "C")
# apply_letter_label(gd, "D")
# apply_letter_label(ge, "E")
colsize!(f.layout, 4, Relative(0.35))
# rowgap!(f.layout, 0)
# rowgap!(f.layout, 2, 10)

f
end

fontsize_theme = Theme(fontsize = 12)
f = with_theme(fontsize_theme) do
    figure_kinematics_info()
end
display(f)

save(joinpath(fig_dir, "fig_kinematics_MI.pdf"), f)



## ------------------------------------------------------------------------



# function figure_kinematics_info_alt()
inch = 96
cm = inch / 2.54
f = Figure(size=(8.5cm, 15cm))


dkn = @pipe df_kine_neur |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_valid_mi)

dt = @pipe df |> 
# @subset(_, :nspikes .> 1000, :peak_mi, :moth .∈ Ref(["2025-02-25", "2025-02-25-1"])) |> 
# leftjoin(_, @subset(df_kine, :peak_mi), on=[:muscle, :moth], renamecols=""=>"_YZ") |> 
@subset(_, :nspikes .> 1000, :peak_valid_mi, :moth .∈ Ref(["2025-02-25", "2025-02-25-1"])) |> 
leftjoin(_, @subset(df_kine, :peak_valid_mi), on=[:muscle, :moth], renamecols=""=>"_YZ") |> 
leftjoin(_, dkn, on=[:neuron, :moth], renamecols=""=>"_XZ") |> 
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

ax = Axis(ga[1,1], xlabel="I(X;Z) - I(X;Y) (bits/s)")
hideydecorations!(ax)
hidespines!(ax, :l, :r, :t)

# Rainclouds sorts using unique(), so create category vector that orders how I want
category = getindex.(Ref(sort_dict), dt.muscle)
inds = sortperm(category, rev=true)
vals = dt.mi_XZ .- dt.mi_XY
# vals = dt.precision_XZ .- dt.precision
rainclouds!(ax, category[inds], vals[inds],
    orientation=:horizontal, plot_boxplots=false,
    cloud_width = 1.5, jitter_width=0.1,
    color=colors[indexin(category[inds], unique(category[inds]))],
)

ax2 = Axis(gb[1,1], xlabel="τ(X;Z) - τ(X;Y) (ms)", xticks=[-10, -5, 0, 5, 10])
hideydecorations!(ax2)
hidespines!(ax2, :l, :r, :t)

# Rainclouds sorts using unique(), so create category vector that orders how I want
category = getindex.(Ref(sort_dict), dt.muscle)
inds = sortperm(category, rev=true)
vals = dt.precision_XZ .- dt.precision
rainclouds!(ax2, category[inds], vals[inds],
    orientation=:horizontal, plot_boxplots=false,
    cloud_width = 1.5, jitter_width=0.1,
    color=colors[indexin(category[inds], unique(category[inds]))],
)


# Muscle labels on left side
# ylims = (0.8, 8.53)
ylims = (0.8, 9.3)
ylims!(ax, ylims)
ylims!(ax2, ylims)
for (i,lab) in enumerate(["All", "Power", "Steering", "DLM", "DVM", "SA", "BA", "AX"])
    # Convert data coordinate i → fraction of axis height
    rel_y = (i - ylims[1]) / (ylims[2]-0.35 - ylims[1])
    fontcol = lab in ["DLM", "DVM", "SA", "BA", "AX"] ? muscle_colors_dict["r" * lowercase(lab)] : :black
    Label(ga[1,1], lab,
        tellheight = false, tellwidth = false,
        halign = :right, valign = rel_y,
        justification=:right,
        padding = (0, 90, 0, 0),
        font = lab == "All" ? :bold : :regular,
        color=fontcol
    )
end

arrows2d!(ax, Point2f(0, 9.1), [[-7,0]], minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5)
arrows2d!(ax, Point2f(0, 9.1), [[+7,0]], minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5)
text!(ax, -10, 8.6, text="More info.\nto muscle(s)", fontsize=10)
text!(ax, 10, 8.6, text="More info.\nto wings", fontsize=10, align=(:right, :bottom))

arrows2d!(ax2, Point2f(0, 9.1), [[-7,0]], minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5)
arrows2d!(ax2, Point2f(0, 9.1), [[+7,0]], minshaftlength=0, tiplength=0.65*8, tipwidth=0.65*14, shaftwidth=2.5)
text!(ax2, -15, 8.6, text="More precise\nto wings", fontsize=10)
text!(ax2, 14, 8.6, text="More precise\nto muscle(s)", fontsize=10, align=(:right, :bottom))

# Inset axes
for (i,lab) in enumerate(["all", "power", "steering", "DLM", "DVM", "SA", "BA", "AX"])
    ax_inset = Axis(f[:,:], 
        width=Relative(0.17), height=Relative(0.17), 
        # halign=0.5, valign=0.155 * i - 0.19, 
        halign=0.5, valign=0.14 * i - 0.2, 
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

apply_letter_label(ga, "A")
apply_letter_label(gb, "B")

colsize!(f.layout, 1, Relative(0.5))

f
# end

# f = figure_kinematics_info_alt()
display(f)