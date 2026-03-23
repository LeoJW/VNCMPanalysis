
"""
This script requires running the first section of main_figures.jl and main_spike_figures.jl to get the necessary imports, dataframes, 
and other things set up and loaded. It's just put here to keep main_figures.jl from becoming 10,000 lines long
"""


## -------------------------------- Figure 3: Main results, neurons are not precise
CairoMakie.activate!()
# GLMakie.activate!()

function figure3()
    f = Figure(size=(1300, 600))

    dfmain = @pipe df |> 
    @subset(_, :peak_valid_mi, :nspikes .> 1000) |> 
    @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi)) |> 
    # @subset(_, :label .== "good") |> 
    @transform(_, :muscle = ifelse.(:single, "single", :muscle))

    df_single_muscles = @subset(df_kine, :peak_mi, :single)
    df_all_muscles = @subset(df_kine, :peak_mi, :muscle .== "all")

    # Diagram of different kinds of MI goes in first panel
    ga = f[1:2,1] = GridLayout()
    img_ax = Axis(ga[1,1], aspect=DataAspect())
    img = load(assetpath(joinpath(fig_dir, "MI_diagram_schematic_vertical.png")))
    image!(img_ax, rotr90(img))
    hidedecorations!(img_ax)
    hidespines!(img_ax)
    # apply_letter_label(ga, "A")

    # Create single GridLayout for both plots
    gb = f[1:2, 2] = GridLayout()

    # MI vs precision for all single muscles
    # Top scatter plot with density margins
    axtop = Axis(gb[1,1])
    ax1 = Axis(gb[2,1], 
        xlabel="I(X;Y) (bits/s)", ylabel="Spike timing precision (ms)",
        # yscale=Makie.log10, yticks=[1, 10],
        yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(10)
    )
    axright = Axis(gb[2,2], 
        # yscale=Makie.log10, yticks=[1, 10],
        yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(10)
    )
    linkxaxes!(ax1, axtop)
    linkyaxes!(ax1, axright)

    # Plot the scatter with density data
    data = dfmain[dfmain.muscle .== "all", :]
    color_dict = Dict("descending" => Makie.wong_colors()[1], "ascending" => Makie.wong_colors()[2], "uncertain" => Makie.wong_colors()[4])

    mi_bins = range(minimum(dfmain.mi[dfmain.mi .> 0,:]), maximum(dfmain.mi[dfmain.mi .> 0,:]), 20)
    prec_bins = range(minimum(data.precision), maximum(data.precision), 20)

    hlines!(ax1, mean(df_all_muscles.precision), linestyle=:dash, color=:black)
    for gdf in groupby(data, :direction)
        label = titlecase(gdf.direction[1])
        scatter!(ax1, gdf.mi, gdf.precision, label=label, color=color_dict[gdf.direction[1]], markersize=12)
    end
    usecol = Makie.wong_colors()[1]
    hist!(axtop, data.mi, bins=mi_bins, normalization=:pdf, color=usecol)
    hist!(axright, data.precision, bins=prec_bins, direction=:x, normalization=:pdf, color=usecol)

    # Add mean value lines
    mean_mi = mean(data.mi)
    mean_precision = mean(data.precision)
    # Calculate histogram heights for MI (top histogram)
    mi_hist = fit(Histogram, data.mi, mi_bins)
    mi_heights = mi_hist.weights ./ (sum(mi_hist.weights) * step(mi_bins))  # Normalize to PDF
    mi_bin_centers = (mi_bins[1:end-1] .+ mi_bins[2:end]) ./ 2
    mean_mi_bin_idx = findmin(abs.(mi_bin_centers .- mean_mi))[2]
    mean_mi_height = mi_heights[mean_mi_bin_idx]
    # Calculate histogram heights for precision (right histogram)
    prec_hist = fit(Histogram, data.precision, prec_bins)
    prec_heights = prec_hist.weights ./ (sum(prec_hist.weights) * (prec_bins[2:end] .- prec_bins[1:end-1]))  # Normalize to PDF
    prec_bin_centers = (prec_bins[1:end-1] .+ prec_bins[2:end]) ./ 2
    mean_prec_bin_idx = findmin(abs.(prec_bin_centers .- mean_precision))[2]
    mean_prec_height = prec_heights[mean_prec_bin_idx]
    # Vertical line on top histogram (axtop) at mean MI value, stopping at histogram height
    lines!(axtop, [mean_mi, mean_mi], [0, mean_mi_height], color=:black, linewidth=2)
    # Horizontal line on right histogram (axright) at mean precision value, stopping at histogram height
    lines!(axright, [0, mean_prec_height], [mean_precision, mean_precision], color=:black, linewidth=2)

    ylims!(axtop, low = 0)
    ylims!(ax1, 0, nothing)
    ylims!(axright, 0, nothing)
    xlims!(axright, low = 0)
    hidedecorations!(axtop, grid = false)
    hidedecorations!(axright, grid = false, minorgrid=false)

    leg = Legend(gb[1, 2], ax1, labelsize=16)
    leg.tellheight = true

    # Bottom raincloud plot
    # axrain = Axis(gb[3,1],
    #     xlabel="Mutual Information (bits/s)"
    # )
    # ax_hasinfo_dist = Axis(gb[3,1], yscale=log10, ylabel="Prob. density")
    # ax_hasinfo_box = Axis(gb[4,1])
    ax_noinfo_dist = Axis(gb[3,1], yscale=log10, ylabel="Prob. density")
    ax_noinfo_box = Axis(gb[4,1], xlabel="I(X;Y) (bits/s)")

    dt = @pipe df |> 
        @subset(_, :muscle .== "all", :nspikes .> 1000) |> 
        @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi)) |> 
        groupby(_, [:moth, :neuron, :muscle]) |> 
        @transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
        @transform(_, :window_select = ifelse.(:peak_off_valid .== "Spike timing info", :peak_valid_mi, :peak_mi)) |> 
        @subset(_, :window_select)
    dt = dt[sortperm(dt.peak_off_valid), :]
    mi_bins = range(0, maximum(dfmain.mi[dfmain.mi .> 0,:])+1, 20)
    # colors = [RGBA(0.5,0.5,0.5,1), Makie.wong_colors()[1]]
    colors = [colorant"#5e3c99", colorant"#e66101"]

    mask = dt.peak_off_valid .== "No spike timing info"
    hist!(ax_noinfo_dist, dt.mi[mask], bins=mi_bins, normalization=:pdf, color=colors[1])
    rainclouds!(ax_noinfo_box, fill("", sum(mask)), dt.mi[mask],
        orientation=:horizontal,
        plot_boxplots=true, clouds=nothing,
        markersize=8, color = colors[1]
    )
    println("$(sum(dt.peak_off_valid .== "Spike timing info")) neurons with timing information")
    println("$(sum(mask)) neurons with no timing information")

    text!(ax_noinfo_dist, 0.6, 0.8, text="No timing information", 
        align=(:center, :center),
        font=:bold, fontsize=18,
        color=colors[1], space=:relative
    )


    # Information between muscles and kinematics I(Y;Z)
    ax_YZ = Axis(gb[2,3],
        xlabel="I(Y;Z) (bits/s)",
        yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(10)
    )
    inset_lims = ((75.4,125.7), (0.5,4))
    ax_YZ_inset = Axis(gb[2,3],
        width=Relative(0.4), height=Relative(0.4),
        halign=0.7, valign=0.8,
        xlabel="I(Y;Z) (bits/s)",
        ylabel="τ(Y;Z) (ms)",
        xticks=[80,100,120], yticks=[1,2,3], limits=inset_lims
    )
    translate!(ax_YZ_inset.blockscene, 0, 0, 150)
    hlines!(ax_YZ, mean(df_all_muscles.precision), linestyle=:dash, color=:black)
    for gdf in groupby(df_single_muscles, :muscle)
        scatter!(ax_YZ, gdf.mi, gdf.precision, color=muscle_colors_dict[gdf.muscle[1]], 
            marker=:rect, label=gdf.muscle[1], markersize=12
        )
        scatter!(ax_YZ_inset, gdf.mi, gdf.precision, color=muscle_colors_dict[gdf.muscle[1]], 
            marker=:rect, label=gdf.muscle[1], markersize=12
        )
    end
    scatter!(ax_YZ, df_all_muscles.mi, df_all_muscles.precision,
        marker=:diamond, color=:black, markersize=12
    )
    border_rect = Rect2(inset_lims[1][1], inset_lims[2][1], 
        inset_lims[1][2] - inset_lims[1][1], inset_lims[2][2] - inset_lims[2][1])
    lines!(ax_YZ, border_rect, color=:grey, linewidth=1)

    muscles = ["LAX", "LBA", "LSA", "LDVM", "LDLM",
    "RAX", "RBA", "RSA", "RDVM", "RDLM"]
    group_color = [MarkerElement(marker=:rect, markersize=20, color=muscle_colors_dict[lowercase(muscle)]) for muscle in muscles]
    group_all = [MarkerElement(marker=:diamond, markersize=20, color=:black)]
    leg_YZ = Legend(gb[1,3], 
        [group_color, group_all], [muscles, ["All muscles"]], [nothing, nothing], 
        nbanks=5, orientation=:vertical, titleposition=:left,
        rowgap=0, colgap=0,
        labelsize=14, titlevisible=nothing
    )
    # leg_YZ = Legend(gb[1, 3], ax_YZ, labelsize=16, orientation=:horizontal, nbanks=2)
    # leg_YZ.tellheight = true

    rowsize!(gb, 3, Relative(0.15))
    rowsize!(gb, 4, Relative(0.08))

    linkxaxes!(ax1, axtop, ax_noinfo_box, ax_noinfo_dist)
    linkyaxes!(ax1, axright, ax_YZ)
    hidexdecorations!(ax_noinfo_dist, grid=false)
    hideydecorations!(ax_noinfo_box)
    xlims!(ax_noinfo_box, 0, nothing)
    
    # Set global gaps and spacing
    colsize!(f.layout, 1, Relative(0.3))
    colgap!(f.layout, -20)
    colsize!(gb, 1, Relative(0.35))
    colgap!(gb, 1, 5)
    rowgap!(gb, 1, 5)
    # colgap!(gb, 2, 40)
    rowgap!(gb, 2, -20)
    rowgap!(gb, 3, 0)
    # rowgap!(gb, 5, 0)

    Label(ga[1,1,TopLeft()], "A",
        fontsize = 30,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right
    )
    Label(gb[1,1,TopLeft()], "B",
        fontsize = 30,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right
    )
    Label(gb[3,1,TopLeft()], "C",
        fontsize = 30,
        font = :bold,
        padding = (0, 0, 0, 0),
        halign = :right
    )
    Label(gb[1,3,TopLeft()], "D",
        fontsize = 30,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right
    )
    display(f)
    f
end

fontsize_theme = Theme(fontsize = 21)
f = with_theme(fontsize_theme) do
    figure3()
end
save(joinpath(fig_dir, "fig3_MI_and_precision.pdf"), f)
display(f)

##

dfmain = @pipe df |> 
@subset(_, :peak_valid_mi, :nspikes .> 1000) |> 
@transform(_, :mi = ifelse.(:mi .< 0, 0, :mi)) |> 
# @subset(_, :label .== "good") |> 
@transform(_, :muscle = ifelse.(:single, "single", :muscle)) |> 
@subset(_, :muscle .== "all") |> 
@transform(_, :eff = :mi ./ :nspikes) |> 
(
AlgebraOfGraphics.data(_) * 
mapping(:mi, :eff, color=:direction) * visual(Scatter)
) |> 
draw(_)



##------------- KS test for whether ascending vs descending from same distribution
dt = @pipe df |> 
@transform(_, :mi = ifelse.(:mi .< 0, 0, :mi)) |> 
@subset(_, :peak_mi, :muscle .== "all", :nspikes .> 1000)

ApproximateTwoSampleKSTest(dt[dt.direction .== "ascending", :mi], dt[dt.direction .== "descending", :mi])

ApproximateTwoSampleKSTest(dt[dt.direction .== "ascending", :precision], dt[dt.direction .== "descending", :precision])

##------------- KS test for whether timing vs no timing significantly different
dt = @pipe df |> 
    @subset(_, :muscle .== "all", :nspikes .> 1000) |> 
    @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi)) |> 
    groupby(_, [:moth, :neuron, :muscle]) |> 
    @transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
    @transform(_, :window_select = ifelse.(:peak_off_valid .== "Spike timing info", :peak_valid_mi, :peak_mi)) |> 
    @subset(_, :window_select)

ApproximateTwoSampleKSTest(dt[dt.peak_off_valid .== "Spike timing info", :mi], dt[dt.peak_off_valid .== "No spike timing info", :mi])
# ApproximateTwoSampleKSTest(dt[dt.peak_off_valid .== "timing", :precision], dt[dt.peak_off_valid .== "no timing", :precision])