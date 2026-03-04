using HDF5
using CSV
using NPZ
using DelimitedFiles
using Statistics
using StatsBase # Mainly for rle
using DSP
using Arrow
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
# wb_duration_thresholds = (50, 70) # ms ldlm
wb_duration_thresholds = (32.5, 100) # ms ldlm

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

function get_phase_dict(moth)
    thismoth = replace(moth, r"-1$" => "_1")
    spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
    labels = npzread(joinpath(data_dir, "..", thismoth * "_labels.npz"))

    # Get neuron keys
    neurons = [k for k in keys(spikes) if labels[k] != 2]
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
    return phase_dict, wblen_dict, muscle_phase_dict
end


function gKJ(θ; μ=pi, γ=0.5, ρ=0, λ=0)
    γbar = (1-ρ^2)/(2*(1-ρ*cos(λ)))
    πprime = γ/γbar
    return πprime / (2*π) * (1 + 2*γbar*(cos(θ-μ)-ρ*cos(λ))/(1 + ρ^2 - 2*ρ*cos(θ-μ-γ))) + (1-πprime)/(2*π)
end


# Get data
mothi = 4
ref_muscle = "ldlm"
histfigs = []
dfp = DataFrame()
for mothi in eachindex(moths)
    phase_dict, wblen_dict, muscle_phase_dict = get_phase_dict(moths[mothi])

    # Get MI of neurons (This df is from main_figures.jl script)
    dfmain = @pipe df |> 
        @subset(_, :peak_mi, :nspikes .> 1000, :muscle .== "all") |> 
        @subset(_, :moth .== replace(moths[mothi], r"_1$" => "-1")) |> 
        @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi))
    sort!(dfmain, [order(:label), order(:mi)])

    n_neur = length(keys(phase_dict))
    f = Figure()
    # Fill good row
    for (i,row) in enumerate(eachrow(@subset(dfmain, :label .== "good")))
        neur = string(round(Int, row.neuron))
        # thisdfc = @subset(dfc, :moth .== replace(moths[mothi], r"_1$" => "-1"), :neuron .== row.neuron)
        thisax = PolarAxis(
            f[1, i],
            title = neur * " " * string(round(row.mi, digits=3))
            # title = neur * " " * string(round(log10(thisdfc[1,:peak_power]), digits=3))
        )
        hiderdecorations!(thisax, grid=false, minorgrid=false)
        hidethetadecorations!(thisax, grid=false, minorgrid=false)
        # density!(thisax, phase_dict[neur] .* 2 * pi)#, normalization=:pdf, bins=100)
        hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[neur]))))
    end
    # Fill MUA row
    for (i,row) in enumerate(eachrow(@subset(dfmain, :label .== "mua")))
        neur = string(round(Int, row.neuron))
        # thisdfc = @subset(dfc, :moth .== replace(moths[mothi], r"_1$" => "-1"), :neuron .== row.neuron)
        thisax = PolarAxis(
            f[2, i],
            title = neur * " " * string(round(row.mi, digits=3))
            # title = neur * " " * string(round(log10(thisdfc[1,:peak_power]), digits=3))
        )
        hiderdecorations!(thisax, grid=false, minorgrid=false)
        hidethetadecorations!(thisax, grid=false, minorgrid=false)
        # density!(thisax, phase_dict[neur] .* 2 * pi)#, normalization=:pdf, bins=100)
        hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=ceil(Int, sqrt(length(phase_dict[neur]))))
    end
    rowgap!(f.layout, 0)
    colgap!(f.layout, 0)
    push!(histfigs, f)
end
display(histfigs[1])

# ------------ Read circular stats from python

dfc = DataFrame(Arrow.Table(joinpath(data_dir, "..", "circular_stats.arrow")))
disallowmissing!(dfc)
vector_cols = ["movm_mu", "kappa", "movm_rvec", "movm_BIC", "mokj_mu", "gamma", "rho", "lam", "mokj_BIC"]
# Clean up types
for col in vector_cols
    dfc[!,col] = [convert(Vector{Float64}, v) for v in dfc[!,col]]
end
transform!(dfc, :moth => ByRow(x -> replace(x, r"_1$" => "-1")) => :moth)

# False Discovery rate correction
α = 0.05
moth_pval_threshold = Dict()
# Example plot for omnibus test
f = Figure()
for (i,dt) in enumerate(groupby(@subset(dfc, :omnibus_p .!= 0), :moth))
    ax = Axis(f[i,1], title=dt.moth[1])
    p = sort(dt.omnibus_p)
    mask = p .<= (collect(1:nrow(dt)) / nrow(dt) * α)
    lastind = findlast(mask)
    moth_pval_threshold[dt.moth[1]] = p[lastind] + (p[lastind+1] - p[lastind])/2
    scatterlines!(ax, collect(1:nrow(dt))[mask], p[mask], color=Makie.wong_colors()[1])
    scatterlines!(ax, collect(1:nrow(dt))[(!).(mask)], p[(!).(mask)], color=:red)
end
f
# Run correction for all tests
test_vars = [:kuiper_p, :watson_p, :rao_p, :omnibus_p]
for test_var in test_vars
    for (i,dt) in enumerate(groupby(dfc, :moth))
        p = sort(dt[!,test_var])
        nonzero = findall(p .!= 0)
        lastind = findlast(p[nonzero] .<= (collect(1:length(p[nonzero])) / length(p[nonzero]) * α))
        lastind = min(length(p[nonzero])-1, lastind)
        moth_pval_threshold = p[nonzero][lastind] + (p[nonzero][lastind+1] - p[nonzero][lastind]) / 2
        dt[!,string(test_var) * "_signif"] = dt[!,test_var] .< moth_pval_threshold
    end
end

# Some summary stats
@pipe dfc |> 
groupby(_, [:moth]) |> 
combine(_, :omnibus_p_signif => mean, :rao_p_signif => mean, :kuiper_p_signif => mean)

# ---------------- Dataframe of each neuron's power at wingbeat frequency
wingbeat_freq_range = [12, 24] # Hz
df_power = DataFrame()
for moth in moths
    thismoth = replace(moth, r"-1$" => "_1")
    spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
    labels = npzread(joinpath(data_dir, "..", thismoth * "_labels.npz"))
    neurons = [k for k in keys(spikes) if labels[k] != 2]

    for neur in neurons
        discrete_vec = zeros(spikes[neur][end]+10)
        discrete_vec[spikes[neur]] .= 1
        pxx = welch_pgram(discrete_vec, fsamp * 10; fs=fsamp)

        fi = findfirst(pxx.freq .> wingbeat_freq_range[1])
        li = findlast(pxx.freq .< wingbeat_freq_range[2])
        peak = findmax(pxx.power[fi:li])
        append!(df_power, DataFrame(
            moth=replace(moth, r"_1$" => "-1"), 
            neuron=parse(Int, neur),
            peak_power=peak[1], # peak power
            peak_freq=pxx.freq[fi:li][peak[2]], # peak freq
            total_power=sum(pxx.power[fi:li]), # total power in range
        ))
    end
end

leftjoin!(dfc, df_power, on=[:moth, :neuron])

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

## ---------------- Circular histograms with Von Mises phases plotted on top

mothi = 3
phase_dict, wblen_dict, muscle_phase_dict = get_phase_dict(moths[mothi])
dfmain = @pipe df |> 
        @subset(_, :peak_mi, :nspikes .> 1000, :muscle .== "all") |> 
        @subset(_, :moth .== replace(moths[mothi], r"_1$" => "-1")) |> 
        @transform(_, :mi = ifelse.(:mi .< 0, 0, :mi))
sort!(dfmain, [order(:label), order(:mi)])

# Max kappa value
max_kappa = maximum(map(maximum, @subset(dfc, :moth .== replace(moths[mothi], r"_1$" => "-1")).gamma))
L, a = 0.1, 0.4

n_neur = length(neurons)
f = Figure()
# Fill good row
for (i,row) in enumerate(eachrow(@subset(dfmain, :label .== "good")))
    neur = string(round(Int, row.neuron))
    thisax = PolarAxis(
        f[1, i],
        title = neur * " " * string(round(row.mi, digits=3))
    )
    hiderdecorations!(thisax, grid=false, minorgrid=false)
    hidethetadecorations!(thisax, grid=false, minorgrid=false)
    # density!(thisax, phase_dict[neur] .* 2 * pi)#, normalization=:pdf, bins=100)
    hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=100, color=:grey)
    
    # # Compute histogram again just to get height 
    h = fit(Histogram, phase_dict[neur] .* 2 * pi)
    max_r = maximum(LinearAlgebra.normalize(h, mode=:pdf).weights)
    dfc_row = @subset(dfc, :neuron .== round(Int, row.neuron), :moth .== row.moth)[1,:]
    for j in 1:dfc_row.mokj_n_clusters
        lines!(thisax, repeat([dfc_row.mokj_mu[j]], 2), [0, dfc_row.gamma[j]], color=Makie.wong_colors()[j])
    end
    # for j in 1:dfc_row.movm_n_clusters
    #     lines!(thisax, repeat([dfc_row.movm_mu[j]], 2), [0, dfc_row.movm_rvec[j] * max_r], color=Makie.wong_colors()[j])
    # end
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
    # density!(thisax, phase_dict[neur] .* 2 * pi)#, normalization=:pdf, bins=100)
    hist!(thisax, phase_dict[neur] .* 2 * pi, normalization=:pdf, bins=100, color=:grey)

    # Compute histogram again just to get height 
    h = fit(Histogram, phase_dict[neur] .* 2 * pi)
    max_r = maximum(LinearAlgebra.normalize(h, mode=:pdf).weights)
    dfc_row = @subset(dfc, :neuron .== round(Int, row.neuron), :moth .== row.moth)[1,:]
    for j in 1:dfc_row.mokj_n_clusters
        lines!(thisax, repeat([dfc_row.mokj_mu[j]], 2), [0, dfc_row.gamma[j]], color=Makie.wong_colors()[j])
    end
    # for j in 1:dfc_row.movm_n_clusters
    #     lines!(thisax, repeat([dfc_row.movm_mu[j]], 2), [0, dfc_row.movm_rvec[j] * max_r], color=Makie.wong_colors()[j])
    # end
end
rowgap!(f.layout, 0)
colgap!(f.layout, 0)
display(f)

##
f = Figure()
for (i,m) in enumerate(moths)
    ax = PolarAxis(f[1,i], title=m)
    for row in eachrow(@subset(dfc, :moth .== replace(m, r"_1$" => "-1")))
        # for j in 1:row.mokj_n_clusters
        #     lines!(ax, [row.mokj_mu[j], row.mokj_mu[j]], [0, row.gamma[j]], color=Makie.wong_colors()[j])
        # end
        for j in 1:row.movm_n_clusters
            lines!(ax, [row.movm_mu[j], row.movm_mu[j]], [0, row.movm_rvec[j]], color=Makie.wong_colors()[j])
        end
    end
end
f

## --------- Condition distribution on wblen
# row = @subset(dfp, :moth .== "2025-02-25", :neuron .== "31") # 31
row = @subset(dfp, :moth .== "2025-02-25-1", :neuron .== "4")
# row = @subset(dfp, :moth .== "2025-03-11", :neuron .== "16") # 11, 16, 
# row = @subset(dfp, :moth .== "2025-03-12-1", :neuron .== "33")

nwb_bin = 4
# wblen_range = [minimum(row.wblen), maximum(row.wblen)]
wblen_range = [58, 70]
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




##

f = Figure()
ax = [Axis(f[i,1]) for i in eachindex(moths)]

for i in eachindex(moths)
    thismoth = replace(moths[i], r"-1$" => "_1")
    spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
    labels = npzread(joinpath(data_dir, "..", thismoth * "_labels.npz"))

    # Get neuron keys
    neurons = [k for k in keys(spikes) if labels[k] != 2]
    muscles = [k for k in keys(spikes) if labels[k] == 2]

    # Extract DLM phase, put all neuron spikes on that phase
    diffvec = diff(spikes[ref_muscle]) ./ fsamp .* 1000 # units of ms
    # mask = (diffvec .> wb_duration_thresholds[1]) .&& (diffvec .< wb_duration_thresholds[2])
    # start_inds = spikes[ref_muscle][findall(vcat(false, mask))]
    # wblen = vcat(diff(start_inds), 10000)
    scatter!(ax[i], spikes[ref_muscle][1:end-1], diffvec)
    hlines!(ax[i], 32.5)
    ylims!(ax[i], 0, 300)
end
f

##

thismoth = replace(moths[2], r"-1$" => "_1")
spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
labels = npzread(joinpath(data_dir, "..", thismoth * "_labels.npz"))
# Get neuron keys
neurons = [k for k in keys(spikes) if labels[k] != 2]
muscles = [k for k in keys(spikes) if labels[k] == 2]

f = Figure()
ax = Axis(f[1,1])
vlines!(ax, spikes["ldlm"] ./ fsamp, ymin=0, ymax=0.5)
vlines!(ax, spikes["18"] ./ fsamp, ymin=0.5, ymax=1.0)
f

##

row = first(eachrow(dfc))

function gKJ(θ; μ=pi, γ=0.5, ρ=0, λ=0)
    γbar = (1-ρ^2)/(2*(1-ρ*cos(λ)))
    πprime = γ/γbar
    return πprime / (2*π) * (1 + 2*γbar*(cos(θ-μ)-ρ*cos(λ))/(1 + ρ^2 - 2*ρ*cos(θ-μ-γ))) + (1-πprime)/(2*π)
end
function gKJ(θ::Vector{Float64}, μ::Vector{Float64}, γ::Vector{Float64}, ρ::Vector{Float64}, λ::Vector{Float64})
    for k in eachindex(μ)
        γbar = (1-ρ^2)/(2*(1-ρ*cos(λ)))
        πprime = γ/γbar

    end
    return πprime / (2*π) * (1 + 2*γbar*(cos(θ-μ)-ρ*cos(λ))/(1 + ρ^2 - 2*ρ*cos(θ-μ-γ))) + (1-πprime)/(2*π)
end


θrange = range(0, 2*π, 1000)

f = Figure()
ax = PolarAxis(f[1,1])
lines!(ax, θrange, gKJ.(θrange; μ=row.mokj_mu[1], γ=row.gamma[1], ρ=row.rho[1], λ=row.lam[1]))
f

## Try out what motor program would look like on polar plot
muscle_colors = [
    "lax" => "#94D63C", "rax" => "#6A992A",
    "lba" => "#AE3FC3", "rba" => "#7D2D8C",
    "lsa" => "#FFBE24", "rsa" => "#E7AC1E",
    "ldvm"=> "#66AFE6", "rdvm"=> "#2A4A78",
    #"ldlm"=> "#E87D7A", "rdlm"=> "#C14434"
]
phase_dict, wblen_dict, muscle_phase_dict = get_phase_dict(moths[1])

f = Figure()
ax = PolarAxis(f[1,1], radius_at_origin=-1)
ax2 = PolarAxis(f[1,2], radius_at_origin=-1)
hiderdecorations!(ax)
hidespines!(ax)
hiderdecorations!(ax2)
hidespines!(ax2)
for i in eachindex(muscle_colors)
    if !(muscle_colors[i][1] in keys(muscle_phase_dict))
        continue
    end
    hist!(ax, muscle_phase_dict[muscle_colors[i][1]] .* 2*pi, 
        bins=90, normalization=:pdf,
        color=(muscle_colors[i][2], 0.5)
    )
    # hist!(ax2, muscle_phase_dict[muscle_colors[i][1]] .* 2*pi, 
    #     bins=90, normalization=:pdf,
    #     color=(muscle_colors[i][2], 0.5)
    # )
end
hist!(ax, phase_dict["54"] .* 2*pi, bins=100, normalization=:pdf, offset=-1)
hist!(ax2, phase_dict["27"] .* 2*pi, bins=100, normalization=:pdf, offset=-1)

lines!(ax, range(0, 2*pi, 1000), zeros(1000), color=:black, linewidth=8)
lines!(ax2, range(0, 2*pi, 1000), zeros(1000), color=:black, linewidth=8)
f

##
# neur = "11"
f = Figure()
ax = [Axis(f[i,1]) for i in eachindex(moths)]
for (i,moth) in enumerate(moths)
    thismoth = replace(moth, r"-1$" => "_1")
    spikes = npzread(joinpath(data_dir, "..", thismoth * "_data.npz"))
    discrete_vec = zeros(spikes["ldlm"][end]+10)
    discrete_vec[spikes["ldlm"]] .= 1
    pxx = welch_pgram(discrete_vec, fsamp * 10; fs=fsamp)
    fi = findfirst(pxx.freq .> 1)
    li = findlast(pxx.freq .< 100)
    lines!(ax[i], pxx.freq[fi:li], pxx.power[fi:li])
end
f