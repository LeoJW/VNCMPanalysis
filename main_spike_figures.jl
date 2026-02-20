using HDF5
using CSV
using NPZ
using DelimitedFiles
using Statistics
using StatsBase # Mainly for rle
using GLM
using CairoMakie
using GLMakie
using Colors
using AlgebraOfGraphics
using DataFrames
using DataFramesMeta
using Pipe
using SavitzkyGolay

include("functions.jl")
include("mainplot_functions.jl")


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

poke_side = Dict(
    "2025-02-25" => "L",
    "2025-02-25-1" => "R",
    "2025-03-11" => "R",
    "2025-03-12-1" => "L",
    "2025-03-20" => "R",
    "2025-03-21" => "R"
)
moths = [
    "2025-02-25",
    "2025-02-25_1",
    "2025-03-11",
    "2025-03-12_1",
    "2025-03-20",
    "2025-03-21"
]
single_muscles = [
    "lax", "lba", "lsa", "ldvm", "ldlm", 
    "rdlm", "rdvm", "rsa", "rba", "rax"
]
muscle_names_dict = Dict(
    "lax-lba-lsa-ldvm-ldlm-rdlm-rdvm-rsa-rba-rax" => "all",
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
muscle_colors = [
    "lax" => "#94D63C", "rax" => "#6A992A",
    "lba" => "#AE3FC3", "rba" => "#7D2D8C",
    "lsa" => "#FFBE24", "rsa" => "#E7AC1E",
    "ldvm"=> "#66AFE6", "rdvm"=> "#2A4A78",
    "ldlm"=> "#E87D7A", "rdlm"=> "#C14434"
]
muscle_colors_dict = Dict(muscle_colors)
fsamp = 30000
# wb_duration_thresholds = (35, 80) # ms ldlm
wb_duration_thresholds = (50, 70) # ms ldlm

function phase_entropy(phases::Vector{<:Real}; nbins=20, normalized=true)
    # Histogram: count spikes per bin
    edges = range(0, 1; length=nbins+1)
    counts = zeros(nbins)
    for φ in phases
        idx = min(Int(floor(φ * nbins)) + 1, nbins)  # clamp phase==1 into last bin
        counts[idx] += 1
    end
    # Convert to probability distribution (skip empty bins — 0*log(0) = 0)
    p = counts ./ sum(counts)
    H = -sum(pᵢ * log2(pᵢ) for pᵢ in p if pᵢ > 0)   # bits
    H_max = log2(nbins)   # entropy of a uniform distribution over nbins
    H_norm = normalized ? H / H_max : H
    return (; H = H_norm, H_raw = H, H_max, nbins)
end


# Get data
mothi = 2
ref_muscle = "ldlm"
histfigs = []
dfp = DataFrame()
# for mothi in eachindex(moths)
    thismoth = replace(moths[mothi], r"-1$" => "_1")
    spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
    labels = npzread(joinpath(data_dir, "..", thismoth * "_labels.npz"))
    bouts = npzread(joinpath(data_dir, "..", thismoth * "_bouts.npz"))

    # Get neuron keys
    neurons = [k for k in keys(spikes) if labels[k] != 2]
    neurons_good = [k for k in keys(spikes) if labels[k] == 1]
    neurons_mua = [k for k in keys(spikes) if labels[k] == 0]
    muscles = [k for k in keys(spikes) if labels[k] == 2]

    # Extract DLM phase, put all neuron spikes on that phase
    diffvec = diff(spikes[ref_muscle]) ./ fsamp .* 1000 # units of ms
    mask = (diffvec .> wb_duration_thresholds[1]) .&& (diffvec .< wb_duration_thresholds[2])
    start_inds = spikes[ref_muscle][findall(vcat(false, mask))]
    wblen = vcat(diff(start_inds), 10000)

    # For each neuron, assign spikes to wingstrokes
    max_duration_thresh_samples = wb_duration_thresholds[2] / 1000 * fsamp
    phase_dict = Dict{String, Vector{Float64}}()
    wblen_dict = Dict{String, Vector{Float64}}()
    for neur in neurons
        wb_assign = searchsortedlast.(Ref(start_inds), spikes[neur])
        # Spikes which are in valid wingbeats (not before first wb or after last, not too long of a wingbeat)
        mask = (wb_assign .!= 0) .&& (wb_assign .<= length(start_inds))
        # New spike vector that's spike index - wingbeat start index / wingbeat length
        phase = (spikes[neur][mask] .- start_inds[wb_assign[mask]]) ./ wblen[wb_assign[mask]]
        length_mask = wblen[wb_assign[mask]] .< max_duration_thresh_samples
        phase = phase[length_mask]
        phase_dict[neur] = phase
        wblen_dict[neur] = wblen[wb_assign[mask]][length_mask] ./ fsamp .* 1000
    end
    muscle_phase_dict = Dict{String, Vector{Float64}}()
    for muscle in muscles
        wb_assign = searchsortedlast.(Ref(start_inds), spikes[muscle])
        # Spikes which are in valid wingbeats (not before first wb or after last, not too long of a wingbeat)
        mask = (wb_assign .!= 0) .&& (wb_assign .<= length(start_inds))
        # New spike vector that's spike index - wingbeat start index / wingbeat length
        phase = (spikes[muscle][mask] .- start_inds[wb_assign[mask]]) ./ wblen[wb_assign[mask]]
        length_mask = wblen[wb_assign[mask]] .< max_duration_thresh_samples
        phase = phase[length_mask]
        muscle_phase_dict[muscle] = phase
    end


    # Get MI of neurons (This df is from main_figures.jl script)
    dfmain = @pipe df |> 
        @subset(_, :peak_mi, :nspikes .> 1000, :muscle .== "all") |> 
        @subset(_, :moth .== replace(moths[mothi], r"_1$" => "-1")) |> 
        @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi))
    sort!(dfmain, [order(:label), order(:mi)])

    n_neur = length(neurons)
    f = Figure()
    # Fill good row
    for (i,row) in enumerate(eachrow(@subset(dfmain, :label .== "good")))
        neur = string(round(Int, row.neuron))
        U = 0.5 * sum(abs.(diff(sort(phase_dict[neur]) .* 2 .* pi) .- (360 / (length(phase_dict[neur])-1))))
        thisax = PolarAxis(
            f[1, i],
            # title = neur * " " * string(round(row.mi, digits=3))
            title = neur * " " * string(round(U, digits=3))
        )
        hiderdecorations!(thisax, grid=false, minorgrid=false)
        hidethetadecorations!(thisax, grid=false, minorgrid=false)
        density!(thisax, phase_dict[neur] .* 2 * pi)#, normalization=:pdf, bins=100)
        hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=100, color=:grey)
        
        C = mean(cos.(phase_dict[neur] .* 2 * pi))
        S = mean(sin.(phase_dict[neur] .* 2 * pi))
        append!(dfp, Dict(
            :moth => row.moth, 
            :neuron => neur,
            :mi => row.mi,
            :label => row.label,
            :direction => row.direction,
            :R => sqrt.(C^2 + S^2),
            :μ => atan(S, C),
            :H => phase_entropy(phase_dict[neur], nbins=100).H,
            :phase => phase_dict[neur],
            :wblen => wblen_dict[neur]
        ))
    end
    # Fill MUA row
    for (i,row) in enumerate(eachrow(@subset(dfmain, :label .== "mua")))
        neur = string(round(Int, row.neuron))
        thisax = PolarAxis(
            f[2, i],
            title = neur * " " * string(round(row.mi, digits=3))
        )
        hiderdecorations!(thisax, grid=false, minorgrid=false)
        hidethetadecorations!(thisax, grid=false, minorgrid=false)
        density!(thisax, phase_dict[neur] .* 2 * pi)#, normalization=:pdf, bins=100)
        hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=100, color=:grey)

        C = mean(cos.(phase_dict[neur] .* 2 * pi))
        S = mean(sin.(phase_dict[neur] .* 2 * pi))
        append!(dfp, Dict(
            :moth => row.moth, 
            :neuron => neur,
            :mi => row.mi,
            :label => row.label,
            :direction => row.direction,
            :R => sqrt.(C^2 + S^2),
            :μ => atan(S, C),
            :H => phase_entropy(phase_dict[neur], nbins=100).H,
            :phase => phase_dict[neur],
            :wblen => wblen_dict[neur]
        ))
    end
    rowgap!(f.layout, 0)
    colgap!(f.layout, 0)
    push!(histfigs, f)
# end
display(f)

## ------------ Read circular stats from python

dfc = DataFrame(Arrow.Table(joinpath(data_dir, "..", "circular_stats.arrow")))
disallowmissing!(dfc)
# Clean up types
dfc.mu = [convert(Vector{Float64}, v) for v in dfc.mu]
dfc.kappa = [convert(Vector{Float64}, v) for v in dfc.kappa]

##
dfcv = @subset(dfc, :mean .!= 0.0)

f = Figure()
for (i,moth) in enumerate(moths)
    ax = PolarAxis(f[1,i], title=moth)
    for row in eachrow(@subset(dfc, :moth .== moth))
        for j in eachindex(row.mu)
            lines!(ax, repeat([row.mu[j]], 2), [0, log(row.kappa[j])+3], color=Makie.wong_colors()[j])
        end
    end
end

## Look at all histograms where n_clusters greater than 
f

##
f = Figure()
ax = Axis(f[1,1], xlabel="H (lower is more phasic)", ylabel="Mutual info to Motor Program (bits/s)")
scatter!(ax, dfp.H, dfp.mi)
f

f = Figure()
ax = Axis(f[1,1], xlabel="R (Rayleigh test for circular uniformity, higher is more phasic)", ylabel="Mutual info to Motor Program (bits/s)")
scatter!(ax, dfp.R, dfp.mi)
f

##
f = Figure()
ax = PolarAxis(f[1,1])
# coldict = Dict("good" => Makie.wong_colors()[1], "mua" => Makie.wong_colors()[2])
# coldict = Dict("descending" => Makie.wong_colors()[1], "ascending" => Makie.wong_colors()[1], "uncertain" => Makie.wong_colors()[3])
coldict = Dict(m => Makie.wong_colors()[i] for (i,m) in enumerate(unique(mothvec)))
for i in eachindex(R)
    # lines!(ax, [μ[i], μ[i]], [0, R[i]], color=log(mi[i]), colorrange=(0, maximum(log.(mi))))
    lines!(ax, [μ[i], μ[i]], [0, R[i]], color=coldict[mothvec[i]])
end
# Colorbar(f[2,1], vertical=false, limits=(0, maximum(log.(mi))))
f

## --------- Condition distribution on wblen
# row = @subset(dfp, :moth .== "2025-02-25", :neuron .== "31") # 31
# row = @subset(dfp, :moth .== "2025-02-25-1", :neuron .== "18")
# row = @subset(dfp, :moth .== "2025-03-11", :neuron .== "16") # 11, 16, 
row = @subset(dfp, :moth .== "2025-03-12-1", :neuron .== "33")

nwb_bin = 5
wblen_range = [minimum(row.wblen), maximum(row.wblen)]
# wblen_range = [minimum(row.wblen), 62]
bins = LinRange(wblen_range[1], wblen_range[2], nwb_bin)

f = Figure()
axwblen = Axis(f[1,1])
axphase = PolarAxis(f[2,1])
color = cgrad(:viridis, nwb_bin-1, categorical=true)
for i in 1:(nwb_bin-1)
    mask = (row.wblen .> bins[i]) .&& (row.wblen .< bins[i+1])
    hist!(axwblen, row.wblen[mask], color=color[i])
    hist!(axphase, row.phase[mask] .* 2 * pi, bins=LinRange(0, 2*pi, 100), normalization=:pdf, color=(color[i], 0.5))
end

f

##


mothi = 2
thismoth = replace(moths[mothi], r"-1$" => "_1")
spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
labels = npzread(joinpath(data_dir, "..", thismoth * "_labels.npz"))


f = Figure()
ax = Axis(f[1,1])
# scatter!(ax, spikes["ldlm"][2:end] ./ fsamp, fsamp ./ diff(spikes["ldlm"]))
# scatter!(ax, spikes["18"][2:end] ./ fsamp, fsamp ./ diff(spikes["18"]))
# ylims!(ax, 0, 1000)
vlines!(ax, spikes["ldlm"] ./ fsamp, ymin=0.5, ymax=1.0)
vlines!(ax, spikes["18"] ./ fsamp, ymin=0.0, ymax=0.5)
f

# 2025-02-25-1 neuron 18
# 2025-03-11 neuron 11


