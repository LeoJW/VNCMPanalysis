using DelimitedFiles
using HDF5
using NPZ
using JSON
using CSV # Read them CSV
using DataFrames
using DataFramesMeta
using StatsBase     # For inverse_rle, countmap
using MultivariateStats # For PCA, suspicious of my PCA function
using LinearAlgebra # For rotations
using DSP # For filters
using BSplineKit # For interpolation
using DataStructures
using GLMakie

include("functions.jl")


function read_events_open_ephys(oedir)
    recording_dir, recording_files = "", []
    for (root, dirs, files) in walkdir(oedir)
        # Once at experiment level, traverse manually from there
        if any(occursin.("experiment", dirs))
            # Get last experiment (that's always the one to use)
            experiment_nums = [parse(Int, s[end]) for s in dirs]
            experiment_dir  = joinpath(root, dirs[argmax(experiment_nums)])
            experiment_contents = readdir(experiment_dir)
            # Take last recording from that experiment. Using tryparse in case last chars aren't numeric
            recording_nums = [tryparse(Int, string(s[end])) for s in experiment_contents]
            recording_nums[isnothing.(recording_nums)] .= 0
            recording_dir = joinpath(experiment_dir, experiment_contents[argmax(recording_nums)])
            recording_files = readdir(recording_dir)
            break
        end
    end
    if isempty(recording_files)
        error("Did not find any experiment directories")
    end
    structure_file = recording_files[findfirst(recording_files .== "structure.oebin")]
    oebin = JSON.parsefile(joinpath(recording_dir, structure_file))
    events_dir = joinpath(recording_dir, "events", oebin["events"][1]["folder_name"])

    times = npzread(joinpath(events_dir, "timestamps.npy"))
    states = npzread(joinpath(events_dir, "states.npy"))
    # full_words = npzread(joinpath(events_dir, "full_words.npy"))
    sample_numbers = npzread(joinpath(events_dir, "sample_numbers.npy"))
    return times, states, sample_numbers
end

"""
Rotate points using transformation from one vector to another
From: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
"""
function rotate_points!(df, a, b, point_names)
    # a is initial vector, b is target vector you want a to point in after rotation
    # Construct rotation matrix
    a = a ./ norm(a)
    b = b ./ norm(b)
    k = cross(a, b)
    Km = [0 -k[3] k[2]; k[3] 0 -k[1]; -k[2] k[1] 0]
    θ = acos(dot(a, b))
    R = I + sin(θ) * Km + (1 - cos(θ)) * Km ^ 2
    # Apply rotation matrix to all 
    for point in point_names
        cols = [Symbol(point * "_x"), Symbol(point * "_y"), Symbol(point * "_z")]
        df[:,cols] = transpose(R * transpose(Matrix(df[:,cols])))
    end
end


function read_kinematics(moth; data_dir = "/Users/leo/Desktop/ResearchPhD/VNCMP/localdata")
    fsamp = 30000
    flap_amp_thresh = 0.4
    flap_duration_thresh = 1000 # samples a bout must last

    # Load open-ephys events, frametimes
    # Special case for 2025-02-25_1 (has rep0 and rep1)
    real_moth = length(split(moth, "_")) >= 3 ? moth[1:end-5] : moth
    # Get open ephys dir as only dir without "sort" or "cam" in name
    moth_dirs = first(walkdir(joinpath(data_dir, real_moth)))[2]
    oedir = moth_dirs[findfirst((!).(occursin.("sort", moth_dirs)) .&& (!).(occursin.("cam", moth_dirs)))]

    times, states, samples = read_events_open_ephys(joinpath(data_dir, real_moth, oedir))
    oeps_times = times[states .== 3]
    samples = samples[states .== 3]

    # Find which frame event to start oeps events; usually the first one is a fluke, caused by arduino startup
    # Special case for rep1 of moth 2025-02-25_1
    if moth == "2025-02-25_1"
        second_start_ind = findlast(diff(oeps_times) .> 1)
        deleteat!(oeps_times, 1:second_start_ind)
        deleteat!(samples, 1:second_start_ind)
    end
    init_times_past_thresh = diff(oeps_times)[1:10] .> 0.01
    start_ind = any(init_times_past_thresh) ? findfirst(init_times_past_thresh) + 1 : 1
    oeps_times = oeps_times[start_ind:end]
    samples = samples[start_ind:end]

    # Read camera timestamps from frametimes.csv files
    frametimes_dir = joinpath(data_dir, moth, "cameras")
    df = [DataFrame() for i in 1:3]
    for i in 0:2
        file = joinpath(frametimes_dir, "frametimes_" * string(i) * ".csv")
        df[i+1] = rename(CSV.read(file, DataFrame), ["frame", "time"])
    end

    # Find initial point where frametimes and open-ephys events agree within some tolerance
    # Prior to this point, frame timestamps will be set from frametimes. 
    # After this point, frame timestamps will be set from open ephys timestamps
    oeps_ind = findfirst(diff(oeps_times[1:20000]) .> 1/350) # Drop to 300 Hz always happens in first 20 seconds
    cam_ind = findfirst(diff(df[1].time[1:20000]) .> 1/350)
    wi = 50
    vec1 = df[1].time[cam_ind-wi:cam_ind+wi] .- df[1].time[1]
    vec2 = oeps_times[oeps_ind-wi:oeps_ind+wi] .- oeps_times[1]
    difference_mat = [x-y for x in vec1, y in vec2]
    row_match = [argmin(abs.(row)) for row in eachrow(difference_mat)]
    # Find where times start to match up 1:1 as longest run length encoding of diff 1
    vals, lens = rle(diff(row_match))
    initial_ind_cam = sum(lens[1:argmax(lens)-1]) + 1
    initial_ind_oeps = row_match[initial_ind_cam]
    # Adjust to indices of full vectors
    initial_ind_cam += cam_ind - wi + 1
    initial_ind_oeps += oeps_ind - wi + 1

    # First part of frame indices from frametimes
    offset_cam_to_oeps = oeps_times[initial_ind_oeps] - df[1].time[initial_ind_cam]
    camera_frametimes = df[1].time[1:initial_ind_cam] .+ offset_cam_to_oeps 

    frame_indices = round.(Int, camera_frametimes .* fsamp)

    # Second part from arduino/open ephys timestamps
    # Find point where desync becomes an issue, mark that point as cutoff
    desync = [
        abs.(df[1].time .- df[2].time) .> 0.0125,
        abs.(df[2].time .- df[3].time) .> 0.0125,
        abs.(df[1].time .- df[3].time) .> 0.0125
    ]
    desync_ind = any(any.(desync)) ? minimum(findfirst.(desync)) : nrow(df[1])
    last_oeps_ind = searchsortedfirst(oeps_times, df[1].time[desync_ind] + offset_cam_to_oeps) - 1
    
    frame_indices = vcat(frame_indices, samples[(initial_ind_oeps+1):last_oeps_ind])
    
    # Read anipose/deeplabcut 3d kinematics up to file where desync happens, go no further
    anipose_dir = joinpath(data_dir, "anipose_VNCMP", moth, "pose-3d")
    kinematics_files = [f for f in readdir(anipose_dir) if occursin(".csv",f)]
    file_inds = [parse(Int, split(f, "_")[1]) for f in kinematics_files]
    sorti = sortperm(file_inds)
    file_inds = file_inds[sorti]
    kinematics_files = kinematics_files[sorti]
    last_file_ind = findlast(file_inds .< desync_ind)
    kdf = DataFrame()
    for file in kinematics_files[1:last_file_ind]
        thisdf = CSV.read(joinpath(anipose_dir, file), DataFrame)
        for col in eachcol(thisdf)
            replace!(col, missing => NaN)
        end
        append!(kdf, thisdf)
    end
    # Keep only frames that were sync'd
    kdf = kdf[1:length(frame_indices), :]
    # Get point names Lhinge, Ltip, etc
    point_cols = setdiff(
        names(kdf),
        ["M_00", "M_01", "M_02", "M_10", "M_11", "M_12", "M_20", "M_21", "M_22",
        "center_0", "center_1", "center_2", "fnum"]
    )
    point_names = [split(x,"_")[1] for x in point_cols[1:6:end]]
    
    # Set x axis along long body axis
    mean_thorax = [mean(col) for col in eachcol(kdf[:, ["Midthorax_" * x for x in ["x","y","z"]]])]
    mean_abdomen = [mean(col) for col in eachcol(kdf[:, ["Midabdomen1_" * x for x in ["x","y","z"]]])]
    rotate_points!(kdf, mean_thorax .- mean_abdomen, [1, 0, 0], point_names)
    # Set y axis from left-right
    front_hinge = [median(kdf[:,"Lhinge_" * d] .- kdf[:,"Rhinge_" * d]) for d in ["x","y","z"]]
    rear_hinge = [median(kdf[:,"Lrearhinge_" * d] .- kdf[:,"Rrearhinge_" * d]) for d in ["x","y","z"]]
    rotate_points!(kdf, front_hinge .+ rear_hinge, [0, 1, 0], point_names)
    # Flip so dorsal side on top
    if kdf.Midthorax_z[1] .< kdf.Lrearhinge_z[1]
        R = [1 0 0; 0 -1 0; 0 0 -1]
        for point in point_names
            cols = [Symbol(point * "_x"), Symbol(point * "_y"), Symbol(point * "_z")]
            kdf[:,cols] = transpose(R * transpose(Matrix(kdf[:,cols])))
        end
    end
    # Center all points at Midthorax
    mean_thorax = [mean(col) for col in eachcol(kdf[:, ["Midthorax_" * x for x in ["x","y","z"]]])]
    for point in point_names
        for (i,d) in enumerate(["_x","_y","_z"])
            kdf[:,Symbol(point * d)] .-= mean_thorax[i]
        end
    end
    
    # Extract wing angles. Three angles: 
    #   Sweep ϕ (angle up and down stroke plane)
    #   Elevation/deviation θ (angle out of plane of stroke plane)
    #   Feathering α (angle between wing plane and stroke plane)
    stroke_plane = [Vector{Vector{Float64}}() for _ in 1:2]
    ϕ = [Vector{Float64}() for _ in 1:2]
    θ = [Vector{Float64}() for _ in 1:2]
    α = [Vector{Float64}() for _ in 1:2]
    Xs_expected = [[0,1,0], [0,-1,0]]
    Ys_expected = [[0,0,-1], [0,0,-1]]
    for (i,side) in enumerate(["L", "R"])
        # Setup, get important quantities
        point = side * "tip"
        cols = [Symbol(point * "_x"), Symbol(point * "_y"), Symbol(point * "_z")]
        hinge2tip = Matrix(kdf[:, cols]) .- Matrix(kdf[:, [side*"hinge_x", side*"hinge_y", side*"hinge_z"]])
        hinge2tip = mapslices(x -> x ./ norm(x), hinge2tip, dims=2) # Make unit vectors
        
        # Roughly grab flapping periods. Used to fit stroke plane better
        filt = digitalfilter(Bandpass(10, 40, fs=300), Butterworth(4))
        heuristic_envelope = abs.(hilbert(filtfilt(filt, hinge2tip[:,3])))
        threshold = maximum(hinge2tip[:,3]) * flap_amp_thresh
        flapping_mask = heuristic_envelope .> threshold
        # Use run length encoding to grab only bouts above a certain duration
        vals, lens = rle(flapping_mask)
        bouts = lens[vals .== 1]
        vals[vals .== 1][bouts .< flap_duration_thresh] .= 0
        flapping_mask = inverse_rle(vals, lens)
        
        # Stroke plane is defined by PC's of wing tip trajectory relative to wing hinge
        pc = MultivariateStats.fit(MultivariateStats.PCA, hinge2tip[flapping_mask,:]', method=:svd)
        pcvec = MultivariateStats.projection(pc)
        # Grab Xs, Ys based on which PCs match expectation
        Xs_match = argmax(abs.([dot(pcvec[j,:], Xs_expected[i]) for j in 1:3]))
        y_inds_avail = setdiff([1,2,3], Xs_match)
        Ys_match = y_inds_avail[argmax(abs.([dot(pcvec[j,:], Ys_expected[i]) for j in y_inds_avail]))]
        Xs = pcvec[Xs_match,:] ./ norm(pcvec[Xs_match,:])
        Ys = pcvec[Ys_match,:] ./ norm(pcvec[Ys_match,:])
        # Enfore sign convention
        # Xs should point out towards tip, Ys down
        Xs_dev = dot(Xs, Xs_expected[i])
        Ys_dev = dot(Ys, Ys_expected[i])
        mult = sign(Xs_dev) == -1 ? -1 : 1
        Xs .*= mult
        mult = sign(Ys_dev) == -1 ? -1 : 1
        Ys .*= mult
        plane_vec = cross(Xs, Ys) # this is a unit vector
        stroke_plane[i] = [Xs,Ys]
        
        # Sweep angle ϕ
        proj = mapslices(x -> plane_vec .* dot(x, plane_vec), hinge2tip, dims=2)
        sweep_vec = hinge2tip .- proj # Hinge to tip, projected onto stroke plane
        sweep_vec = mapslices(x -> x ./ norm(x), sweep_vec, dims=2)
        ϕ[i] = [acos(dot(sweep_vec[i,:], Ys)) - pi/2 for i in 1:size(sweep_vec)[1]]
        # Deviation θ
        θ[i] = [pi/2 - acos(dot(row, plane_vec)) for row in eachrow(hinge2tip)]
        # Feathering α
        tip_cols = [side * "tip_" * d for d in ["x", "y", "z"]]
        rear_cols = [side * "rear_" * d for d in ["x", "y", "z"]]
        mid_cols = [side * "midlead_" * d for d in ["x", "y", "z"]]
        vec1 = Matrix(kdf[:, rear_cols]) .- Matrix(kdf[:, tip_cols])
        vec2 = Matrix(kdf[:, mid_cols]) .- Matrix(kdf[:, tip_cols]) # x product of these should point up
        # Make unit vectors
        vec1 = mapslices(x -> x ./ norm(x), vec1, dims=2) 
        vec2 = mapslices(x -> x ./ norm(x), vec2, dims=2) 
        if side == "R"
            vec1, vec2 = vec2, vec1
        end
        wing_plane = reduce(hcat, [cross(vec1[i,:], vec2[i,:]) for i in axes(vec1, 1)])'
        α[i] = [pi/2 - acos(dot(row, plane_vec)) for row in eachrow(wing_plane)]
    end

    # Match up kinematics to actual time points, resample/interpolate
    interp_frames = round.(Int, range(frame_indices[1], step=fsamp/300, length=length(frame_indices)))
    # Set up output dict
    data_dict = Dict{String, Vector{}}()
    data_dict["index"] = interp_frames
    # Add angles
    for (i,side) in enumerate(["L", "R"])
        data_dict[side * "phi"] = interpolate(frame_indices, ϕ[i], BSplineOrder(4)).(interp_frames)
        data_dict[side * "theta"] = interpolate(frame_indices, θ[i], BSplineOrder(4)).(interp_frames)
        data_dict[side * "alpha"] = interpolate(frame_indices, α[i], BSplineOrder(4)).(interp_frames)
    end
    # Add x y z point data
    for point in point_names
        for d in ["_x", "_y", "_z"]
            col = point * d
            data_dict[col] = interpolate(frame_indices, kdf[:,col], BSplineOrder(4)).(interp_frames)
        end
    end

    # return ϕ, θ, α, kdf, stroke_plane, frame_indices, frame_times
    return data_dict
end
