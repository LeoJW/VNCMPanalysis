"""
Originally provided online by Intan
Modified (hastily) by Leo Wood 09/2023 
Because what kind of insane monster writes shit to global variables in the workspace?!?!
"""

#= Define data structure for data channels =#
mutable struct ChannelStruct
    native_channel_name::String
    custom_channel_name::String
    native_order::Int16
    custom_order::Int16
    board_stream::Int16
    chip_channel::Int16
    port_name::String
    port_prefix::String
    port_number::Int16
    electrode_impedance_magnitude::Float32
    electrode_impedance_phase::Float32
end

#= Define data structure for spike trigger settings =#
mutable struct SpikeTriggerStruct
    voltage_trigger_mode::Int16
    voltage_threshold::Int16
    digital_trigger_channel::Int16
    digital_edge_polarity::Int16
end

#= Define data structure for frequency parameters for RHD system =#
mutable struct RHDFrequencyParametersStruct
    amplifier_sample_rate::Float64
    aux_input_sample_rate::Float64
    supply_voltage_sample_rate::Float64
    board_adc_sample_rate::Float64
    board_dig_in_sample_rate::Float64
    desired_dsp_cutoff_frequency::Float64
    actual_dsp_cutoff_frequency::Float64
    dsp_enabled::Bool
    desired_lower_bandwidth::Float64
    actual_lower_bandwidth::Float64
    desired_upper_bandwidth::Float64
    actual_upper_bandwidth::Float64
    notch_filter_frequency::Float64
    desired_impedance_test_frequency::Float64
    actual_impedance_test_frequency::Float64
end

#= Define data structure for frequency parameters for RHS system =#
mutable struct RHSFrequencyParametersStruct
    amplifier_sample_rate::Float64
    board_adc_sample_rate::Float64
    board_dig_in_sample_rate::Float64
    desired_dsp_cutoff_frequency::Float64
    actual_dsp_cutoff_frequency::Float64
    dsp_enabled::Bool
    desired_lower_bandwidth::Float64
    desired_lower_settle_bandwidth::Float64
    actual_lower_bandwidth::Float64
    actual_lower_settle_bandwidth::Float64
    desired_upper_bandwidth::Float64
    actual_upper_bandwidth::Float64
    notch_filter_frequency::Float64
    desired_impedance_test_frequency::Float64
    actual_impedance_test_frequency::Float64
end

#= Define data structure for stimulation parameters =#
mutable struct StimParametersStruct
    stim_step_size::Float64
    charge_recovery_current_limit::Float64
    charge_recovery_target_voltage::Float64
    amp_settle_mode::Int16
    charge_recovery_mode::Int16
end

# Quick and dirty modified function to just get N of amplifier and ADC 
# Could be WAY faster but this is the fastest for me to write
function read_rhd_size(filenamestring)
    fid = open(filenamestring, "r")
    filesize = stat(filenamestring).size
    #= Check 'magic number' at beginning of file to make sure this is an Intan Technologies RHD2000 data file =#
    magicnumber = read(fid, UInt32)
    if magicnumber != 0xc6912702
        error("Unrecognized file type.")
    end
    #= Read version number =#
    datafilemainversionnumber = read(fid, Int16)
    datafilesecondaryversionnumber = read(fid, Int16)
    if datafilemainversionnumber == 1
        numsamplesperdatablock = 60
    else
        numsamplesperdatablock = 128
    end
    samplerate = read(fid, Float32)
    dspenabled = read(fid, Int16)
    actualdspcutofffrequency = read(fid, Float32)
    actuallowerbandwidth = read(fid, Float32)
    actualupperbandwidth = read(fid, Float32)
    desireddspcutofffrequency = read(fid, Float32)
    desiredlowerbandwidth = read(fid, Float32)
    desiredupperbandwidth = read(fid, Float32)
    #= This tells us is a software 50/60 Hz notch filter was enabled during the data acquisition =#
    notchfiltermode = read(fid, Int16)
    notchfilterfrequency = 0
    if notchfiltermode == 1
        notchfilterfrequency = 50
    elseif notchfiltermode == 2
        notchfilterfrequency = 60
    end
    read(fid, Float32)
    read(fid, Float32)
    #= Place notes in array of Strings =#
    notes = [readQString(fid), readQString(fid), readQString(fid)]
    #= If data file is from GUI v1.1 or later, see if temperature sensor data was saved =#
    numtempsensorchannels = 0
    if (datafilemainversionnumber == 1 && datafilesecondaryversionnumber >= 1) || (datafilemainversionnumber > 1)
        numtempsensorchannels = read(fid, Int16)
    end
    #= If data file is from GUI v1.3 or later, load eval board mode =#
    if (datafilemainversionnumber == 1 && datafilesecondaryversionnumber >= 3) || (datafilemainversionnumber > 1)
        read(fid, Int16)
    end
    #= If data file is from v2.0 or later (Intan Recording Controller), load name of digital reference channel =#
    if datafilemainversionnumber > 1
        readQString(fid)
    end
    amplifierindex = 1
    auxinputindex = 1
    supplyvoltageindex = 1
    boardadcindex = 1
    boarddiginindex = 1
    boarddigoutindex = 1
    #= Read signal summary from data file header =#
    numberofsignalgroups = read(fid, Int16)
    for _ = 1:numberofsignalgroups
        readQString(fid)
        readQString(fid)
        signalgroupenabled = read(fid, Int16)
        signalgroupnumchannels = read(fid, Int16)
        read(fid, Int16)
        if (signalgroupnumchannels > 0) && (signalgroupenabled > 0)
            for _ = 1:signalgroupnumchannels
                readQString(fid)
                readQString(fid)
                read(fid, Int16)
                read(fid, Int16)
                signaltype = read(fid, Int16)
                channelenabled = read(fid, Int16)
                read(fid, Int16)
                read(fid, Int16)
                read(fid, Int16)
                read(fid, Int16)
                read(fid, Int16)
                read(fid, Int16)
                read(fid, Float32)
                read(fid, Float32)
                if channelenabled > 0
                    if signaltype == 0
                        amplifierindex = amplifierindex + 1
                    elseif signaltype == 1
                        auxinputindex = auxinputindex + 1
                    elseif signaltype == 2
                        supplyvoltageindex = supplyvoltageindex + 1
                    elseif signaltype == 3
                        boardadcindex = boardadcindex + 1
                    elseif signaltype == 4
                        boarddiginindex = boarddiginindex + 1
                    elseif signaltype == 5
                        boarddigoutindex = boarddigoutindex + 1
                    else
                        error("Unknown channel type")
                    end
                end
            end
        end
    end
    #= Summarize contents of data file =#
    numamplifierchannels = amplifierindex - 1
    numauxinputchannels = auxinputindex - 1
    numsupplyvoltagechannels = supplyvoltageindex - 1
    numboardadcchannels = boardadcindex - 1
    numboarddiginchannels = boarddiginindex - 1
    numboarddigoutchannels = boarddigoutindex - 1

    #= Each data block contains numSamplesPerDataBlock amplifier samples =#
    bytesperblock = numsamplesperdatablock * 4
    bytesperblock = bytesperblock + numsamplesperdatablock * 2 * numamplifierchannels
    #= Auxiliary inputs are sampled 4x slower than amplifiers =#
    bytesperblock = bytesperblock + (numsamplesperdatablock / 4) * 2 * numauxinputchannels
    #= Supply voltage is sampled once per data block =#
    bytesperblock = bytesperblock + 1 * 2 * numsupplyvoltagechannels
    #= Board analog inputs are sampled at same rate as amplifiers =#
    bytesperblock = bytesperblock + numsamplesperdatablock * 2 * numboardadcchannels
    #= Board digital inputs are sampled at same rate as amplifiers =#
    if numboarddiginchannels > 0
        bytesperblock = bytesperblock + numsamplesperdatablock * 2
    end
    #= Board digital outputs are sampled at same rate as amplifiers =#
    if numboarddigoutchannels > 0
        bytesperblock = bytesperblock + numsamplesperdatablock * 2
    end
    #= Temp sensor is sampled once per data block =#
    if numtempsensorchannels > 0
        bytesperblock = bytesperblock + 1 * 2 * numtempsensorchannels
    end
    #= How many data blocks remain in this file? =#
    datapresent = 0
    bytesremaining = filesize - position(fid)
    if bytesremaining > 0
        datapresent = 1
    end
    close(fid)
    numamplifiersamples = Int(numsamplesperdatablock * Int(bytesremaining / bytesperblock))
    return numamplifiersamples
end

#= Read the given file as .rhd format =#
function read_data_rhd(filenamestring; 
    read_amplifier=true,
    read_adc=false,
    read_aux=false,
    read_temp_sensor=false,
    read_supply_voltage=false,
    read_digital_in=false,
    read_digital_out=false,
    verbose=false)

    #= Reads Intan Technologies RHD2000 data file generated by evaluation board GUI. Data are returned in ... =#

    if verbose
        println("Attempting to open file")
        start = time()
    end
    fid = open(filenamestring, "r")
    filesize = stat(filenamestring).size

    #= Check 'magic number' at beginning of file to make sure this is an Intan Technologies RHD2000 data file =#
    magicnumber = read(fid, UInt32)
    if magicnumber != 0xc6912702
        error("Unrecognized file type.")
    end

    #= Read version number =#
    datafilemainversionnumber = read(fid, Int16)
    datafilesecondaryversionnumber = read(fid, Int16)

    if verbose
        println("\nReading Intan Technologies RHD2000 Data File, Version ", datafilemainversionnumber, ".", datafilesecondaryversionnumber)
    end

    if datafilemainversionnumber == 1
        numsamplesperdatablock = 60
    else
        numsamplesperdatablock = 128
    end

    #= Read information of sampling rate and amplifier frequency settings =#
    samplerate = read(fid, Float32)
    dspenabled = read(fid, Int16)
    actualdspcutofffrequency = read(fid, Float32)
    actuallowerbandwidth = read(fid, Float32)
    actualupperbandwidth = read(fid, Float32)

    desireddspcutofffrequency = read(fid, Float32)
    desiredlowerbandwidth = read(fid, Float32)
    desiredupperbandwidth = read(fid, Float32)

    #= This tells us is a software 50/60 Hz notch filter was enabled during the data acquisition =#
    notchfiltermode = read(fid, Int16)
    notchfilterfrequency = 0
    if notchfiltermode == 1
        notchfilterfrequency = 50
    elseif notchfiltermode == 2
        notchfilterfrequency = 60
    end

    desiredimpedancetestfrequency = read(fid, Float32)
    actualimpedancetestfrequency = read(fid, Float32)

    #= Place notes in array of Strings =#
    notes = [readQString(fid), readQString(fid), readQString(fid)]

    #= If data file is from GUI v1.1 or later, see if temperature sensor data was saved =#
    numtempsensorchannels = 0
    if (datafilemainversionnumber == 1 && datafilesecondaryversionnumber >= 1) || (datafilemainversionnumber > 1)
        numtempsensorchannels = read(fid, Int16)
    end

    #= If data file is from GUI v1.3 or later, load eval board mode =#
    evalboardmode = 0
    if (datafilemainversionnumber == 1 && datafilesecondaryversionnumber >= 3) || (datafilemainversionnumber > 1)
        evalboardmode = read(fid, Int16)
    end

    #= If data file is from v2.0 or later (Intan Recording Controller), load name of digital reference channel =#
    if datafilemainversionnumber > 1
        referencechannel = readQString(fid)
    end

    #= Place frequency-related information in data structure =#
    frequencyparameters = RHDFrequencyParametersStruct(
        samplerate,
        samplerate/4,
        samplerate / numsamplesperdatablock,
        samplerate,
        samplerate,
        desireddspcutofffrequency,
        actualdspcutofffrequency,
        dspenabled,
        desiredlowerbandwidth,
        actuallowerbandwidth,
        desiredupperbandwidth,
        actualupperbandwidth,
        notchfilterfrequency,
        desiredimpedancetestfrequency,
        actualimpedancetestfrequency)

    spiketriggers = SpikeTriggerStruct[]

    amplifierchannels = ChannelStruct[]
    auxinputchannels = ChannelStruct[]
    supplyvoltagechannels = ChannelStruct[]
    boardadcchannels = ChannelStruct[]
    boarddiginchannels = ChannelStruct[]
    boarddigoutchannels = ChannelStruct[]

    amplifierindex = 1
    auxinputindex = 1
    supplyvoltageindex = 1
    boardadcindex = 1
    boarddiginindex = 1
    boarddigoutindex = 1

    #= Read signal summary from data file header =#
    numberofsignalgroups = read(fid, Int16)

    for signalgroup = 1 : numberofsignalgroups

        signalgroupname = readQString(fid)
        signalgroupprefix = readQString(fid)
        signalgroupenabled = read(fid, Int16)
        signalgroupnumchannels = read(fid, Int16)
        signalgroupnumampchannels = read(fid, Int16)

        if (signalgroupnumchannels > 0) && (signalgroupenabled > 0)
            for signalchannel = 1:signalgroupnumchannels
                newtriggerchannel = SpikeTriggerStruct(0, 0, 0, 0)

                newchannel = ChannelStruct("", "", 0, 0, 0, 0, "", "", 0, 0.0, 0.0)

                newchannel.port_name = signalgroupname
                newchannel.port_prefix = signalgroupprefix
                newchannel.port_number = signalgroup

                newchannel.native_channel_name = readQString(fid)
                newchannel.custom_channel_name = readQString(fid)
                newchannel.native_order = read(fid, Int16)
                newchannel.custom_order = read(fid, Int16)
                signaltype = read(fid, Int16)
                channelenabled = read(fid, Int16)
                newchannel.chip_channel = read(fid, Int16)
                newchannel.board_stream = read(fid, Int16)
                newtriggerchannel.voltage_trigger_mode = read(fid, Int16)
                newtriggerchannel.voltage_threshold = read(fid, Int16)
                newtriggerchannel.digital_trigger_channel = read(fid, Int16)
                newtriggerchannel.digital_edge_polarity = read(fid, Int16)
                newchannel.electrode_impedance_magnitude = read(fid, Float32)
                newchannel.electrode_impedance_phase = read(fid, Float32)

                if channelenabled > 0
                    if signaltype == 0
                        push!(amplifierchannels, newchannel)
                        push!(spiketriggers, newtriggerchannel)
                        amplifierindex = amplifierindex + 1
                    elseif signaltype == 1
                        push!(auxinputchannels, newchannel)
                        auxinputindex = auxinputindex + 1
                    elseif signaltype == 2
                        push!(supplyvoltagechannels, newchannel)
                        supplyvoltageindex = supplyvoltageindex + 1
                    elseif signaltype == 3
                        push!(boardadcchannels, newchannel)
                        boardadcindex = boardadcindex + 1
                    elseif signaltype == 4
                        push!(boarddiginchannels, newchannel)
                        boarddiginindex = boarddiginindex + 1
                    elseif signaltype == 5
                        push!(boarddigoutchannels, newchannel)
                        boarddigoutindex = boarddigoutindex + 1
                    else
                        error("Unknown channel type")
                    end
                end
            end
        end
    end

    #= Summarize contents of data file =#
    numamplifierchannels = amplifierindex - 1
    numauxinputchannels = auxinputindex - 1
    numsupplyvoltagechannels = supplyvoltageindex - 1
    numboardadcchannels = boardadcindex - 1
    numboarddiginchannels = boarddiginindex - 1
    numboarddigoutchannels = boarddigoutindex - 1

    if verbose
        println("Found ", numamplifierchannels, " amplifier channel", plural(numamplifierchannels))
        println("Found ", numauxinputchannels, " auxiliary input channel", plural(numauxinputchannels))
        println("Found ", numsupplyvoltagechannels, " supply voltage channel", plural(numsupplyvoltagechannels))
        println("Found ", numboardadcchannels, " board ADC channel", plural(numboardadcchannels))
        println("Found ", numboarddiginchannels, " board digital input channel", plural(numboarddiginchannels))
        println("Found ", numboarddigoutchannels, " board digital output channel", plural(numboarddigoutchannels))
        println("Found ", numtempsensorchannels, " temperature sensor channel", plural(numtempsensorchannels), ".\n")
    end

    #= Determine how many samples the data file contains =#

    #= Each data block contains numSamplesPerDataBlock amplifier samples =#
    bytesperblock = numsamplesperdatablock * 4
    bytesperblock = bytesperblock + numsamplesperdatablock * 2 * numamplifierchannels
    #= Auxiliary inputs are sampled 4x slower than amplifiers =#
    bytesperblock = bytesperblock + (numsamplesperdatablock / 4) * 2 * numauxinputchannels
    #= Supply voltage is sampled once per data block =#
    bytesperblock = bytesperblock + 1 * 2 * numsupplyvoltagechannels
    #= Board analog inputs are sampled at same rate as amplifiers =#
    bytesperblock = bytesperblock + numsamplesperdatablock * 2 * numboardadcchannels
    #= Board digital inputs are sampled at same rate as amplifiers =#
    if numboarddiginchannels > 0
        bytesperblock = bytesperblock + numsamplesperdatablock * 2
    end
    #= Board digital outputs are sampled at same rate as amplifiers =#
    if numboarddigoutchannels > 0
        bytesperblock = bytesperblock + numsamplesperdatablock * 2
    end
    #= Temp sensor is sampled once per data block =#
    if numtempsensorchannels > 0
        bytesperblock = bytesperblock + 1 * 2 * numtempsensorchannels
    end

    #= How many data blocks remain in this file? =#
    datapresent = 0
    bytesremaining = filesize - position(fid)
    if bytesremaining > 0
        datapresent = 1
    end

    numdatablocks = Int(bytesremaining / bytesperblock)

    numamplifiersamples = Int(numsamplesperdatablock * numdatablocks)
    numauxinputsamples = Int((numsamplesperdatablock / 4) * numdatablocks)
    numsupplyvoltagesamples = Int(1 * numdatablocks)
    numboardadcsamples = Int(numsamplesperdatablock * numdatablocks)
    numboarddiginsamples = Int(numsamplesperdatablock * numdatablocks)
    numboarddigoutsamples = Int(numsamplesperdatablock * numdatablocks)

    recordtime = numamplifiersamples / samplerate

    if verbose && datapresent > 0
        println("File contains $(recordtime) seconds of data.  Amplifiers were sampled at $(samplerate/1000) kS/s.")
    elseif verbose
        println("Header file contains no data.  Amplifiers were sampled at $(samplerate/1000) kS/s.")
    end

    if datapresent > 0
        #= Pre-allocate memory for data =#
        if verbose
            println("Allocating memory for data...\n")
        end

        if (datafilemainversionnumber == 1 && datafilesecondaryversionnumber >= 2) || (datafilemainversionnumber > 1)
            tamplifier = zeros(Int32, 1, numamplifiersamples)
        else
            tamplifier = zeros(UInt32, 1, numamplifiersamples)
        end

        amplifierdata = read_amplifier ? zeros(UInt16, numamplifierchannels, numamplifiersamples) : nothing
        auxinputdata = read_aux ? zeros(Float64, numauxinputchannels, numauxinputsamples) : nothing
        supplyvoltagedata = read_supply_voltage ? zeros(Float64, numsupplyvoltagechannels, numsupplyvoltagesamples) : nothing
        tempsensordata = read_temp_sensor ? zeros(Float64, numtempsensorchannels, numsupplyvoltagesamples) : nothing
        boardadcdata = read_adc ? zeros(Float64, numboardadcchannels, numboardadcsamples) : nothing
        boarddigindata = read_digital_in ? zeros(Int16, numboarddiginchannels, numboarddiginsamples) : nothing
        boarddiginraw = read_digital_in ? Vector{UInt16}(undef, numboarddiginsamples) : nothing
        boarddigoutdata = read_digital_out ? zeros(Int16, numboarddigoutchannels, numboarddigoutsamples) : nothing
        boarddigoutraw = read_digital_out ? Vector{UInt16}(undef, numboarddigoutsamples) : nothing

        #= Read sampled data from file =#
        if verbose
            println("Reading data from file...\n")
        end

        amplifierindex = Int(1)
        auxinputindex = Int(1)
        supplyvoltageindex = Int(1)
        boardadcindex = Int(1)
        boarddiginindex = Int(1)
        boarddigoutindex = Int(1)

        printincrement = 10
        percentdone = printincrement

        for i = 1 : numdatablocks
            #= In version 1.2, we moved from saving timestamps as unsigned
            integers to signed integers to accomodate negative (adjusted)
            timestamps for pretrigger data =#

            if (datafilemainversionnumber == 1 && datafilesecondaryversionnumber >= 2) || (datafilemainversionnumber > 1)
                tamplifier[amplifierindex:(amplifierindex + numsamplesperdatablock - 1)] = reinterpret(Int32, read(fid, numsamplesperdatablock * 4))
            else
                tamplifier[amplifierindex:(amplifierindex + numsamplesperdatablock - 1)] = reinterpret(UInt32, read(fid, numsamplesperdatablock * 4))
            end

            if read_amplifier && numamplifierchannels > 0
                amplifierdata[:, amplifierindex:(amplifierindex + numsamplesperdatablock - 1)] = permutedims(reshape(reinterpret(UInt16, read(fid, numsamplesperdatablock * numamplifierchannels * 2)), numsamplesperdatablock, numamplifierchannels))
            elseif numamplifierchannels > 0
                skip(fid, numsamplesperdatablock * numamplifierchannels * 2)
            end

            if read_aux && numauxinputchannels > 0
                auxinputdata[:, auxinputindex:(auxinputindex + Int(numsamplesperdatablock / 4) - 1)] = permutedims(reshape(reinterpret(UInt16, read(fid, Int(numsamplesperdatablock * numauxinputchannels * 2 / 4))), Int(numsamplesperdatablock / 4), numauxinputchannels))
            elseif numauxinputchannels > 0
                skip(fid, Int(numsamplesperdatablock * numauxinputchannels * 2 / 4))
            end

            if read_supply_voltage && numsupplyvoltagechannels > 0
                supplyvoltagedata[:, supplyvoltageindex] = reinterpret(UInt16, read(fid, numsupplyvoltagechannels * 2))
            elseif numsupplyvoltagechannels > 0
                skip(fid, numsupplyvoltagechannels * 2)
            end

            if read_temp_sensor && numtempsensorchannels > 0
                tempsensordata[:, supplyvoltageindex] = reinterpret(UInt16, read(fid, numtempsensorchannels * 2))
            elseif numtempsensorchannels > 0
                skip(fid, numtempsensorchannels * 2)
            end

            if read_adc && numboardadcchannels > 0
                boardadcdata[:, boardadcindex:(boardadcindex + numsamplesperdatablock - 1)] = permutedims(reshape(reinterpret(UInt16, read(fid, numsamplesperdatablock * numboardadcchannels * 2)), numsamplesperdatablock, numboardadcchannels))
            elseif numboardadcchannels > 0
                skip(fid, numsamplesperdatablock * numboardadcchannels * 2)
            end
            
            if read_digital_in && numboarddiginchannels > 0
                boarddiginraw[boarddiginindex:(boarddiginindex + numsamplesperdatablock - 1)] = reinterpret(UInt16, read(fid, numsamplesperdatablock * 2))
            elseif numboarddiginchannels > 0
                read(fid, numsamplesperdatablock * 2)
            end

            if read_digital_out && numboarddigoutchannels > 0
                boarddigoutraw[boarddigoutindex:(boarddigoutindex + numsamplesperdatablock - 1)] = reinterpret(UInt16, read(fid, numsamplesperdatablock * 2))
            elseif numboarddigoutchannels > 0
                skip(fid, numsamplesperdatablock * 2)
            end

            amplifierindex = amplifierindex + numsamplesperdatablock
            auxinputindex = Int(auxinputindex + (numsamplesperdatablock / 4))
            supplyvoltageindex = supplyvoltageindex + 1
            boardadcindex = boardadcindex + numsamplesperdatablock
            boarddiginindex = boarddiginindex + numsamplesperdatablock
            boarddigoutindex = boarddigoutindex + numsamplesperdatablock

            fractiondone = 100 * (i / numdatablocks)
            if verbose && fractiondone >= percentdone
                println(percentdone, "% done...")
                percentdone = percentdone + printincrement
            end
        end

        #= Make sure we have read exactly the right amount of data =#
        bytesremaining = filesize - position(fid)
        if bytesremaining != 0
            error("Error: End of file not reached.")
        end

    end

    #= Close data file =#
    close(fid)

    datapresent = datapresent > 0

    if datapresent
        if verbose
            println("Parsing data...\n")
        end

        #= Extract digital input channels to separate variables =#
        if read_digital_in
            for i = 1 : numboarddiginchannels
                mask = 2^boarddiginchannels[i].native_order
                boarddigindata[i,:] = (x -> (x > 0 ? 1 : 0)).(boarddiginraw .& mask)
            end
        end

        #= Extract digital output channels to separate variables =#
        if read_digital_out
            for i = 1 : numboarddigoutchannels
                mask = 2^boarddigoutchannels[i].native_order
                boarddigoutdata[i,:] = (x -> (x > 0 ? 1 : 0)).(boarddigoutraw .& mask)
            end
        end

        #= Scale voltage levels appropriately =#
        amplifierdata = read_amplifier ? 0.195 .* (amplifierdata .- 32768) : nothing # units = microvolts
        auxinputdata = read_aux ? 37.4e-6 .* auxinputdata : nothing# units = volts
        supplyvoltagedata = read_supply_voltage ? 74.8e-6 .* supplyvoltagedata : nothing# units = volts

        if read_adc
            if evalboardmode == 1
                boardadcdata = 152.59e-6 .* (boardadcdata .- 32768) # units = volts
            elseif evalboardmode == 13 # Intan Recording Controller
                boardadcdata = 312.5e-6 .* (boardadcdata .- 32768) # units = volts
            else
                boardadcdata = 50.354e-6 .* boardadcdata # units = volts
            end
        end
        tempsensordata = read_temp_sensor ? tempsensordata ./ 100 : nothing # units = deg C

        #= Check for gaps in timestamps =#
        numgaps = sum(diff(tamplifier, dims=2)[1, :] .!= 1)
        if verbose && numgaps == 0
            println("No missing timestamps in data.")
        elseif verbose
            println("Warning: ", numgaps, " gaps in timestamp data found. Time scale will not be uniform!")
        end

        #= Scale time steps (units = seconds) =#
        tamp = Array{Float64,2}(undef, 1, length(tamplifier))
        tamp[:] = tamplifier[:] ./ samplerate
        tempvar = length(tamp)
        if read_aux
            tauxinput = Array{Float64,2}(undef, 1, Int(tempvar/4))
            tauxinput[1, :] = tamp[1, 1:4:tempvar]
        end
        if read_supply_voltage || read_temp_sensor
            tsupplyvoltage = Array{Float64,2}(undef, 1, Int(tempvar/numsamplesperdatablock))
            tsupplyvoltage[1, :] = tamp[1:numsamplesperdatablock:tempvar]
        end

        #= If the software notch filter was selected during the recording, apply the same notch filter to amplifier data here =#

        if notchfilterfrequency > 0 && datafilemainversionnumber < 3 && read_amplifier
            if verbose
                println("Applying notch filter...\n")
            end
            printincrement = 10
            percentdone = printincrement
            for i = 1:numamplifierchannels
                amplifierdata[i,:] = notchfilter(amplifierdata[i,:], samplerate, notchfilterfrequency, 10)
                fractiondone = 100 * (i / numamplifierchannels)
                if verbose && fractiondone >= percentdone
                    println(percentdone, "% done...")
                    percentdone = percentdone + printincrement
                end
            end
        end


    end

    """
    # (DON'T) Move variables to base workspace
    # Just move variables to dict and return that

    Some notes:
    amplifier, ADC, and digital time are all the same. Pasing those as just "time"
    supply voltage and temp sensory times are the same, pass those as "time_sensor"
    AUX is its own thing, gets its own time

    For now ignoring spiketriggers
    """
    datadict = Dict{String, Any}()
    # Times
    if read_amplifier || read_adc || read_digital_in || read_digital_out
        datadict["time"] = tamp
    end
    if read_supply_voltage || read_temp_sensor
        datadict["time_sensor"] = tsupplyvoltage
    end
    # Amplifier
    if read_amplifier
        datadict["amplifier"] = amplifierdata
    end
    # ADC
    if read_adc && numboardadcchannels > 0
        datadict["adc_channels"] = boardadcchannels
        datadict["adc"] = datapresent ? boardadcdata : nothing
    end
    # AUX
    if read_aux
        datadict["aux_channels"] = auxinputchannels
        datadict["time_aux"] = tauxinput
        datadict["aux"] = datapresent ? auxinputdata : nothing
    end
    # Supply voltage
    if read_supply_voltage && numsupplyvoltagechannels > 0
        datadict["supply_voltage_channels"] = supplyvoltagechannels
        datadict["supply_voltage"] = datapresent ? supplyvoltagedata : nothing
    end
    # Temp sensor
    if read_temp_sensor && numtempsensorchannels > 0
        datadict["temp_sensor"] = datapresent ? tempsensordata : nothing
    end
    # Digital in
    if read_digital_in && numboarddiginchannels > 0
        datadict["digital_in_channels"] = boarddiginchannels
        datadict["digital_in"] = datapresent ? boarddigindata : nothing
    end
    # Digital out
    if read_digital_out && numboarddigoutchannels > 0
        datadict["digital_out_channels"] = boarddigoutchannels
        datadict["digital_out"] = datapresent ? boarddigoutdata : nothing
    end
    # Extra stuff
    datadict["notes"] = notes
    datadict["frequency_parameters"] = frequencyparameters
    if datafilemainversionnumber > 1
        datadict["reference_channel"] = referencechannel
    end
    datadict["N"] = numamplifiersamples

    if verbose
        elapsed = time() - start
        println("Done! Elapsed time: $(elapsed) seconds")
        if datapresent > 0
            println("Extracted data returned as dict.")
        else
            println("Extracted waveform information returned as dict.")
        end
    end
    
    return datadict
end


#= Read the given file as .rhd format =#
function read_data_rhs(filenamestring)

    #= Reads Intan Technologies RHS2000 data file generated by Stimulation/Recording Controller. =#
    println("Attempting to open file")

    start = time()

    fid = open(filenamestring, "r")
    filesize = stat(filenamestring).size

    #= Check 'magic number' at beginning of file to make sure this is an Intan Technologies RHS2000 data file =#
    magicnumber = read(fid, UInt32)
    if magicnumber != 0xd69127ac
        error("Unrecognized file type.")
    end

    #= Read version number =#
    datafilemainversionnumber = read(fid, Int16)
    datafilesecondaryversionnumber = read(fid, Int16)

    println("\nReading Intan Technologies RHS2000 Data File, Version ", datafilemainversionnumber, ".", datafilesecondaryversionnumber)

    numsamplesperdatablock = 128

    #= Read information of sampling rate and amplifier frequency settings. =#
    samplerate = read(fid, Float32)
    dspenabled = read(fid, Int16)
    actualdspcutofffrequency = read(fid, Float32)
    actuallowerbandwidth = read(fid, Float32)
    actuallowersettlebandwidth = read(fid, Float32)
    actualupperbandwidth = read(fid, Float32)

    desireddspcutofffrequency = read(fid, Float32)
    desiredlowerbandwidth = read(fid, Float32)
    desiredlowersettlebandwidth = read(fid, Float32)
    desiredupperbandwidth = read(fid, Float32)

    #= This tells us if a software 50/60 Hz notch filter was enabled during the data acquisition =#
    notchfiltermode = read(fid, Int16)
    notchfilterfrequency = 0
    if notchfiltermode == 1
        notchfilterfrequency = 50
    elseif notchfiltermode == 2
        notchfilterfrequency = 60
    end

    desiredimpedancetestfrequency = read(fid, Float32)
    actualimpedancetestfrequency = read(fid, Float32)

    ampsettlemode = read(fid, Int16)
    chargerecoverymode = read(fid, Int16)

    stimstepsize = read(fid, Float32)
    chargerecoverycurrentlimit = read(fid, Float32)
    chargerecoverytargetvoltage = read(fid, Float32)

    #= Place notes in array of Strings =#
    notes = [readQString(fid), readQString(fid), readQString(fid)]

    #= See if dc amplifier data was saved =#
    dcampdatasaved = read(fid, Int16)

    #= Load eval board mode =#
    evalboardmode = read(fid, Int16)

    referencechannel = readQString(fid)

    #= Place frequency-related information in data structure. =#
    frequencyparameters = RHSFrequencyParametersStruct(samplerate,
                                                    samplerate,
                                                    samplerate,
                                                    desireddspcutofffrequency,
                                                    actualdspcutofffrequency,
                                                    dspenabled,
                                                    desiredlowerbandwidth,
                                                    desiredlowersettlebandwidth,
                                                    actuallowerbandwidth,
                                                    actuallowersettlebandwidth,
                                                    desiredupperbandwidth,
                                                    actualupperbandwidth,
                                                    notchfilterfrequency,
                                                    desiredimpedancetestfrequency,
                                                    actualimpedancetestfrequency)

    stimparameters = StimParametersStruct(stimstepsize,
                                        chargerecoverycurrentlimit,
                                        chargerecoverytargetvoltage,
                                        ampsettlemode,
                                        chargerecoverymode)

    spiketriggers = SpikeTriggerStruct[]

    #= Create structure arrays for each type of data channel =#

    amplifierchannels = ChannelStruct[]
    boardadcchannels = ChannelStruct[]
    boarddacchannels = ChannelStruct[]
    boarddiginchannels = ChannelStruct[]
    boarddigoutchannels = ChannelStruct[]

    amplifierindex = 1
    boardadcindex = 1
    boarddacindex = 1
    boarddiginindex = 1
    boarddigoutindex = 1

    #= Read signal summary from data file header =#
    numberofsignalgroups = read(fid, Int16)

    for signalgroup = 1 : numberofsignalgroups
        signalgroupname = readQString(fid)
        signalgroupprefix = readQString(fid)
        signalgroupenabled = read(fid, Int16)
        signalgroupnumchannels = read(fid, Int16)
        signalgroupnumampchannels = read(fid, Int16)

        if (signalgroupnumchannels > 0) && (signalgroupenabled > 0)

            for signalchannel = 1 : signalgroupnumchannels

                newtriggerchannel = SpikeTriggerStruct(0, 0, 0, 0)
                newchannel = ChannelStruct("", "", 0, 0, 0, 0, "", "", 0, 0.0, 0.0)

                newchannel.port_name = signalgroupname
                newchannel.port_prefix = signalgroupprefix
                newchannel.port_number = signalgroup

                newchannel.native_channel_name = readQString(fid)
                newchannel.custom_channel_name = readQString(fid)
                newchannel.native_order = read(fid, Int16)
                newchannel.custom_order = read(fid, Int16)
                signaltype = read(fid, Int16)
                channelenabled = read(fid, Int16)
                newchannel.chip_channel = read(fid, Int16)
                read(fid, Int16) # ignore command_stream
                newchannel.board_stream = read(fid, Int16)
                newtriggerchannel.voltage_trigger_mode = read(fid, Int16)
                newtriggerchannel.voltage_threshold = read(fid, Int16)
                newtriggerchannel.digital_trigger_channel = read(fid, Int16)
                newtriggerchannel.digital_edge_polarity = read(fid, Int16)
                newchannel.electrode_impedance_magnitude = read(fid, Float32)
                newchannel.electrode_impedance_phase = read(fid, Float32)

                if channelenabled > 0
                    if signaltype == 0
                        push!(amplifierchannels, newchannel)
                        push!(spiketriggers, newtriggerchannel)
                        amplifierindex = amplifierindex + 1
                    elseif signaltype == 1
                        # aux inputs; not used in RHS2000 system
                    elseif signaltype == 2
                        # supply voltage; not used in RHS2000 system
                    elseif signaltype == 3
                        push!(boardadcchannels, newchannel)
                        boardadcindex = boardadcindex + 1
                    elseif signaltype == 4
                        push!(boarddacchannels, newchannel)
                        boarddacindex = boarddacindex + 1
                    elseif signaltype == 5
                        push!(boarddiginchannels, newchannel)
                        boarddiginindex = boarddiginindex + 1
                    elseif signaltype == 6
                        push!(boarddigoutchannels, newchannel)
                        boarddigoutindex = boarddigoutindex + 1
                    else
                        error("Unknown channel type")
                    end
                end
            end
        end
    end

    #= Summarize contents of data file =#
    numamplifierchannels = amplifierindex - 1
    numboardadcchannels = boardadcindex - 1
    numboarddacchannels = boarddacindex - 1
    numboarddiginchannels = boarddiginindex - 1
    numboarddigoutchannels = boarddigoutindex - 1

    println("Found ", numamplifierchannels, " amplifier channel", plural(numamplifierchannels))
    if dcampdatasaved != 0
        println("Found ", numamplifierchannels, " DC amplifier channel", plural(numamplifierchannels))
    end
    println("Found ", numboardadcchannels, " board ADC channel", plural(numboardadcchannels))
    println("Found ", numboarddacchannels, " board DAC channel", plural(numboarddacchannels))
    println("Found ", numboarddiginchannels, " board digital input channel", plural(numboarddiginchannels))
    println("Found ", numboarddigoutchannels, " board digital output channel", plural(numboarddigoutchannels), ".\n")

    #= Determine how many samples the data file contains =#

    #= Each data block contains numsamplesperdatablock amplifier samples =#
    bytesperblock = numsamplesperdatablock * 4  # timestamp data
    if dcampdatasaved != 0
        bytesperblock = bytesperblock + numsamplesperdatablock * (2 + 2 + 2) * numamplifierchannels
    else
        bytesperblock = bytesperblock + numsamplesperdatablock * (2 + 2) * numamplifierchannels
    end

    #= Board analog inputs are sampled at same rate as amplifiers =#
    bytesperblock = bytesperblock + numsamplesperdatablock * 2 * numboardadcchannels
    #= Board analog outputs are sampled at same rate as amplifiers =#
    bytesperblock = bytesperblock + numsamplesperdatablock * 2 * numboarddacchannels
    #= Board digital inputs are sampled at same rate as amplifiers =#
    if numboarddiginchannels > 0
        bytesperblock = bytesperblock + numsamplesperdatablock * 2
    end
    #= Board digital outputs are sampled at same rate as amplifiers =#
    if numboarddigoutchannels > 0
        bytesperblock = bytesperblock + numsamplesperdatablock * 2
    end

    #= How many data blocks remain in this file? =#
    datapresent = 0
    bytesremaining = filesize - position(fid)
    if bytesremaining > 0
        datapresent = 1
    end

    numdatablocks = Int(bytesremaining / bytesperblock)

    numamplifiersamples = Int(numsamplesperdatablock * numdatablocks)
    numboardadcsamples = Int(numsamplesperdatablock * numdatablocks)
    numboarddacsamples = Int(numsamplesperdatablock * numdatablocks)
    numboarddiginsamples = Int(numsamplesperdatablock * numdatablocks)
    numboarddigoutsamples = Int(numsamplesperdatablock * numdatablocks)

    recordtime = numamplifiersamples / samplerate

    if datapresent > 0
        println("File contains $(recordtime) seconds of data.  Amplifiers were sampled at $(samplerate/1000) kS/s.")
    else
        println("Header file contains no data.  Amplifiers were sampled at $(samplerate/1000) kS/s.")
    end

    if datapresent > 0
        #= Pre-allocate memory for data =#
        println("Allocating memory for data...\n")

        t = zeros(Int32, 1, numamplifiersamples)

        amplifierdata = zeros(UInt16, numamplifierchannels, numamplifiersamples)
        if dcampdatasaved != 0
            dcamplifierdata = zeros(UInt16, numamplifierchannels, numamplifiersamples)
        end
        stimdata = zeros(UInt16, numamplifierchannels, numamplifiersamples)
        ampsettledata = zeros(UInt16, numamplifierchannels, numamplifiersamples)
        chargerecoverydata = zeros(UInt16, numamplifierchannels, numamplifiersamples)
        compliancelimitdata = zeros(UInt16, numamplifierchannels, numamplifiersamples)
        boardadcdata = zeros(Float64, numboardadcchannels, numboardadcsamples)
        boarddacdata = zeros(Float64, numboarddacchannels, numboarddacsamples)
        boarddigindata = zeros(Int16, numboarddiginchannels, numboarddiginsamples)
        boarddiginraw = Vector{UInt16}(undef, numboarddiginsamples)
        boarddigoutdata = zeros(Int16, numboarddigoutchannels, numboarddigoutsamples)
        boarddigoutraw = Vector{UInt16}(undef, numboarddigoutsamples)

        #= Read sampled data from file =#
        println("Reading data from file...\n")

        amplifierindex = 1
        boardadcindex = 1
        boarddacindex = 1
        boarddiginindex = 1
        boarddigoutindex = 1

        printincrement = 10
        percentdone = printincrement

        for i = 1 : numdatablocks
            t[amplifierindex:(amplifierindex + numsamplesperdatablock - 1)] = reinterpret(Int32, read(fid, numsamplesperdatablock * 4))
            if numamplifierchannels > 0
                amplifierdata[:, amplifierindex:(amplifierindex + numsamplesperdatablock - 1)] = permutedims(reshape(reinterpret(UInt16, read(fid, numsamplesperdatablock * numamplifierchannels * 2)), numsamplesperdatablock, numamplifierchannels))
                if dcampdatasaved != 0
                    dcamplifierdata[:, amplifierindex:(amplifierindex + numsamplesperdatablock - 1)] = permutedims(reshape(reinterpret(UInt16, read(fid, numsamplesperdatablock * numamplifierchannels * 2)), numsamplesperdatablock, numamplifierchannels))
                end
                stimdata[:, amplifierindex:(amplifierindex + numsamplesperdatablock - 1)] = permutedims(reshape(reinterpret(UInt16, read(fid, numsamplesperdatablock * numamplifierchannels * 2)), numsamplesperdatablock, numamplifierchannels))
            end

            if numboardadcchannels > 0
                boardadcdata[:, boardadcindex:(boardadcindex + numsamplesperdatablock - 1)] = permutedims(reshape(reinterpret(UInt16, read(fid, numsamplesperdatablock * numboardadcchannels * 2)), numsamplesperdatablock, numboardadcchannels))
            end
            if numboarddacchannels > 0
                boarddacdata[:, boarddacindex:(boarddacindex + numsamplesperdatablock - 1)] = permutedims(reshape(reinterpret(UInt16, read(fid, numsamplesperdatablock * numboarddacchannels * 2)), numsamplesperdatablock, numboarddacchannels))
            end

            if numboarddiginchannels > 0
                boarddiginraw[boarddiginindex:(boarddiginindex + numsamplesperdatablock - 1)] = reinterpret(UInt16, read(fid, numsamplesperdatablock * 2))
            end
            if numboarddigoutchannels > 0
                boarddigoutraw[boarddigoutindex:(boarddigoutindex + numsamplesperdatablock - 1)] = reinterpret(UInt16, read(fid, numsamplesperdatablock * 2))
            end

            amplifierindex = amplifierindex + numsamplesperdatablock
            boardadcindex = boardadcindex + numsamplesperdatablock
            boarddacindex = boarddacindex + numsamplesperdatablock
            boarddiginindex = boarddiginindex + numsamplesperdatablock
            boarddigoutindex = boarddigoutindex + numsamplesperdatablock

            fractiondone = 100 * (i / numdatablocks)
            if fractiondone >= percentdone
                println(percentdone, "% done...")
                percentdone = percentdone + printincrement
            end
        end

        #= Make sure we have read exactly the right amount of data. =#
        bytesremaining = filesize - position(fid)
        if bytesremaining != 0
            error("Error: End of file not reached.")
        end
    end

    #= Close data file =#
    close(fid)

    if datapresent > 0
        println("Parsing data...\n")

        #= Extract digital input channels to separate variables =#
        for i = 1 : numboarddiginchannels
            mask = 2^boarddiginchannels[i].native_order
            boarddigindata[i,:] = (x -> (x > 0 ? 1 : 0)).(boarddiginraw .& mask)
        end

        #= Extract digital output channels to separate variables =#
        for i = 1 : numboarddigoutchannels
            mask = 2^boarddigoutchannels[i].native_order
            boarddigoutdata[i,:] = (x -> (x > 0 ? 1 : 0)).(boarddigoutraw .& mask)
        end

        #= Scale voltage levels appropriately =#
        amplifierdata = 0.195 .* (amplifierdata .- 32768) # units = microvolts
        if (dcampdatasaved != 0)
            dcamplifierdata = -0.01923 .* (dcamplifierdata .- 512) # units = volts
        end

        compliancelimitdata = (x -> ((x >= 2^15) ? 1 : 0)).(stimdata)
        stimdata = stimdata - (compliancelimitdata .* 2^15)
        chargerecoverydata = (x -> ((x >= 2^14) ? 1 : 0)).(stimdata)
        stimdata = stimdata - (chargerecoverydata .* 2^14)
        ampsettledata = (x -> ((x >= 2^13) ? 1 : 0)).(stimdata)
        stimdata = stimdata - (ampsettledata .* 2^13)
        stimpolarity = (x -> ((x >= 2^8) ? 1 : 0)).(stimdata)
        stimdata = stimdata - (stimpolarity .* 2^8)
        stimpolarity = 1 .- (2 .* stimpolarity) # convert (0 = pos, 1 = neg) to +/- 1
        stimdata = stimdata .* stimpolarity
        stimdata = stimdata .* (stimparameters.stim_step_size / 1.0e-6) # units = microamps
        #stimdata = stimparameters.stim_step_size .* stimdata / 1.0e-6 # units = microamps
        boardadcdata = 312.5e-6 .* (boardadcdata .- 32768) # units = volts
        boarddacdata = 312.5e-6 .* (boarddacdata .- 32768) # units = volts

        #= Check for gaps in timestamps =#
        numgaps = sum(diff(t, dims=2)[1, :] .!= 1)
        if numgaps == 0
            println("No missing timestamps in data.")
        else
            println("Warning: ", numgaps, " gaps in timestamp data found. Time scale will not be uniform!")
        end

        #= Scale time steps (units = seconds). =#
        t = t ./ samplerate

        #= If the software notch filter was selected during the recording, apply the same notch filter to amplifier data here =#

        if notchfilterfrequency > 0 && datafilemainversionnumber < 3
            println("Applying notch filter...\n")
            printincrement = 10
            percentdone = printincrement
            for i = 1:numamplifierchannels
                amplifierdata[i,:] = notchfilter(amplifierdata[i,:], samplerate, notchfilterfrequency, 10)
                fractiondone = 100 * (i / numamplifierchannels)
                if fractiondone >= percentdone
                    println(percentdone, "% done...")
                    percentdone = percentdone + printincrement
                end
            end
        end

    end


    # Move variables to base workspace
    global notes = notes
    global frequency_parameters = frequencyparameters
    global stim_parameters = stimparameters
    if datafilemainversionnumber > 1
        global reference_channel = referencechannel
    end

    if numamplifierchannels > 0
        global amplifier_channels = amplifierchannels
        if datapresent > 0
            global amplifier_data = amplifierdata
            if dcampdatasaved != 0
                global dc_amplifier_data = dcamplifierdata
            end
            global stim_data = stimdata
            global amp_settle_data = ampsettledata
            global charge_recovery_data = chargerecoverydata
            global compliance_limit_data = compliancelimitdata
            global t = t
        end
        global spike_triggers = spiketriggers
    end

    if numboardadcchannels > 0
        global board_adc_channels = boardadcchannels
        if datapresent > 0
            global board_adc_data = boardadcdata
        end
    end


    if numboarddacchannels > 0
        global board_dac_channels = boarddacchannels
        if datapresent > 0
            global board_dac_data = boarddacdata
        end
    end

    if numboarddiginchannels > 0
        global board_dig_in_channels = boarddiginchannels
        if datapresent > 0
            global board_dig_in_data = boarddigindata
        end
    end

    if numboarddigoutchannels > 0
        global board_dig_out_channels = boarddigoutchannels
        if datapresent > 0
            global board_dig_out_data = boarddigoutdata
        end
    end

    elapsed = time() - start

    println("Done! Elapsed time: $(elapsed) seconds")
    if datapresent > 0
        println("Extracted data are now available in the Julia workspace.")
    else
        println("Extracted waveform information is now available in the Julia workspace.")
    end

end


function readQString(fid)
    # a = readQString

    #= Read Qt style QString. The first 32-bit unsigned number indicates
    the length of the string (in bytes). If this number equals 0xffffffff,
    the string is null =#

    a = ""
    length = read(fid, UInt32)
    if length == 0xffffffff
        return
    end

    #= convert length from bytes to 16-bit Unicode words =#
    length = length / 2
    for i = 1:length
        thisChar = Char(read(fid, UInt16))
        a = a * thisChar
    end
    return a
end


function plural(n)
    # s = plural(n)
    # Utility function to optionally pluralize words based on the value of n
    if n == 1
        s = ""
    else
        s = "s"
    end
    return s
end


function notchfilter(in, fsample, fnotch, bandwidth)

    tstep = 1 / Float64(fsample)
    fc = fnotch * tstep

    l = length(in)

    # Calculate IIR filter parameters
    d = exp(-2 * pi * (bandwidth / 2) * tstep)
    b = (1 + d * d) * cos(2 * pi * fc)
    a0 = 1
    a1 = -b
    a2 = d * d
    a = (1 + d * d) / 2
    b0 = 1
    b1 = -2 * cos(2 * pi * fc)
    b2 = 1

    out = Vector{Float64}(undef, length(in))
    out[1] = in[1]
    out[2] = in[2]

    #= (If filtering a continuous data stream, change out[1] and out[2] to the previous final two values of out.) =#
    #= Run filter =#
    for k = 3:l
        out[k] = (a*b2*in[k-2] + a*b1*in[k-1] + a*b0*in[k] - a2*out[k-2] - a1*out[k-1])/a0
    end

    return out
end


function read_data(filename)
    if last(filename, 4) == ".rhd"
        read_data_rhd(filename)
    elseif last(filename, 4) == ".rhs"
        read_data_rhs(filename)
    else
        error("The given file doesn't appear to be an Intan .rhd or .rhs file")
    end
end

function read_data(filename, channel)
    println("Second argument used")
    println(channel)
end


function test_intan_reader()
	println("Successfully accessed IntanReader")
end
