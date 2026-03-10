"""
Kinematics figure


This script requires running the first section of main_figures.jl and main_spike_figures.jl to get the necessary imports, dataframes, 
and other things set up and loaded. It's just put here to keep main_figures.jl from becoming 10,000 lines long
"""

CairoMakie.activate!()
# GLMakie.activate!()


function figure_kinematics_info()
f = Figure(size=(700, 500))

color_dict = Dict("descending" => Makie.wong_colors()[1], "ascending" => Makie.wong_colors()[2], "uncertain" => Makie.wong_colors()[4])

dkn = @pipe df_kine_neur |> 
groupby(_, [:moth, :neuron]) |> 
@transform(_, :peak_off_valid = ifelse.(findfirst(:peak_mi) .!= findfirst(:peak_valid_mi), "No spike timing info", "Spike timing info")) |> 
@subset(_, :peak_mi)

dt = @pipe df |> 
@subset(_, :peak_mi, :moth .∈ Ref(["2025-02-25", "2025-02-25-1"])) |> 
leftjoin(_, @subset(df_kine, :peak_mi), on=[:muscle, :moth], renamecols=""=>"_YZ") |> 
leftjoin(_, dkn, on=[:neuron, :moth], renamecols=""=>"_XZ") |> 
@transform(_, :mi_XY = :mi) |> 
@transform(_, :n2m_loss = :mi_XY .- :mi_XZ) |> 
@transform(_, :m2k_loss = :mi_YZ .- :mi_XZ) #|> 
# stack(_, [:mi_XY, :mi_XZ, :mi_YZ])

ga = f[1,1:3] = GridLayout()

# AX, BA, SA, DVM, DLM panels
ax_muscle = []
for (i,muscle) in enumerate(["ax", "ba", "sa", "dvm", "dlm"])
    if muscle .== "ax"
        thisax = Axis(ga[1,i], 
            xlabel="I(X;" * uppercase(muscle) * ") (bits/s)", ylabel="I(X;Z) (bits/s)",
            limits=((0, nothing), (0, nothing))
        )
    else
        thisax = Axis(ga[1,i], xlabel="I(X;" * uppercase(muscle) * ") (bits/s)", limits=((0, nothing), (0, nothing)))
        hideydecorations!(thisax, ticks=false, grid=false, minorgrid=false, minorticks=false)
    end
    push!(ax_muscle, thisax)
    ablines!(thisax, 0, 1, color=:black)
    mask = [x[2:end] .== muscle for x in dt.muscle]
    scatter!(thisax, dt[mask, :mi_XY], dt[mask,:mi_XZ], color=:black)
    # for gdf in groupby(dt[mask,:], :direction)
    #     scatter!(thisax, gdf.mi_XY, gdf.mi_XZ, color=color_dict[gdf.direction[1]])
    # end

    # Inset axis
    ax_inset = Axis(ga[1,i], width=Relative(0.4), height=Relative(0.4), halign=0.1, valign=0.9, aspect=DataAspect())
    img = load(assetpath(joinpath(fig_dir, "muscles_$(uppercase(muscle)).png")))
    image!(ax_inset, rotr90(img))
    hidedecorations!(ax_inset)
    hidespines!(ax_inset)
end
linkaxes!(ax_muscle...)

# All muscles panels
gb = f[2,1] = GridLayout()

ax_all = Axis(gb[1,1], xlabel="I(X;Y) (bits/s)", ylabel="I(X;Z) (bits/s)", title="All muscles")
mask = dt.muscle .== "all"
ablines!(ax_all, 0, 1, color=:black)
scatter!(ax_all, dt[mask, :mi_XY], dt[mask, :mi_XZ], color=:black)
xlims!(ax_all, low=0)
ylims!(ax_all, low=0)

gc = f[2,2] = GridLayout()
# lastax = Axis(gc[1,1])
# mask = dt.muscle .== "all"
# scatter!(lastax, dt[mask, :mi_XY], dt[mask, :mi_XZ] .- dt[mask, :mi_XY])
# xlims!(lastax, low=0)
ax_prec_all = Axis(gc[1,1])



gd = f[2,3] = GridLayout()
ax_window = Axis(gd[1,1])

apply_letter_label(ga, "A")
apply_letter_label(gb, "B")
apply_letter_label(gc, "C")
apply_letter_label(gd, "D")
rowsize!(f.layout, 2, Relative(0.65))

f
end

f = figure_kinematics_info()
display(f)

save(joinpath(fig_dir, "fig_kinematics_MI.pdf"), f)