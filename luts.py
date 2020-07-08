# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for setting up the Look-up tables for amplitude,
frequency and phase used by Vivace.
"""

import numpy as np

import utils


def get_LUT_values(vips):
    """
    Constructs matrices of amplitude and frequency/phase values from the pulse definitions.
    The matrices are indexed by port number, and will be the basis for the LUTs on the board.
    If all pulses have fixed amp/freq/phase values, each value is only present once in the list.
    If a sweep pulse is defined, the list will be a complete chronological sequence of values,
    so that the board will only ever need to step once in the LUT between pulses.
    Return two matrices: one for the amplitude scale values and one for tuples of frequency and phase values.
    """
    # Sample pulses are not relevant here, so take them out of the definition list
    pulse_definitions_no_samples = [pulse for pulse in vips.pulse_definitions if 'Sample' not in pulse]

    # Divide the pulses by the port they're on
    port_timelines = [[] for _ in range(8)]
    for pulse in pulse_definitions_no_samples:
        p_port_idx = pulse['Port'] - 1
        port_timelines[p_port_idx].append(pulse)

    amp_values = [[] for _ in range(8)]
    freq_phase_values = [[] for _ in range(8)]
    port_carrier_changes = [[] for _ in range(8)]

    for port_idx, timeline in enumerate(port_timelines):
        # Don't do anything for ports without definitions
        if len(timeline) < 1:
            continue

        carrier_changes = []

        latest_fp1 = None
        latest_fp2 = None
        latest_change = None

        # Step through each iteration to put together the LUTs in the order the values will be needed.
        for i in range(vips.iterations):

            # Add any left over values from previous iteration
            if i > 0:
                if latest_fp2 is None:
                    latest_fp2 = (0, 0)
                if latest_fp1 is None:
                    latest_fp1 = (0, 0)
                carrier_changes.append((latest_change,
                                        (latest_fp1[0], latest_fp1[1]),
                                        (latest_fp2[0], latest_fp2[1])))

            # Reset for the new iteration
            latest_fp1 = None
            latest_fp2 = None
            # The first carrier change will happen at the first pulse
            first_pulse_start = timeline[0]['Time']
            latest_change = utils.get_absolute_time(vips, first_pulse_start[0], first_pulse_start[1], i)
            reference_times = {}

            for pulse in timeline:
                # Get some necessary parameters from the pulse's definition.
                p_amp, p_freq, p_phase = utils.get_amp_freq_phase(pulse, i)
                time = pulse['Time']
                abs_time = utils.get_absolute_time(vips, time[0], time[1], i)

                # Save the amplitude value if it is new
                if p_amp not in amp_values[port_idx]:
                    amp_values[port_idx].append(p_amp)

                # If the pulse isn't using a carrier wave, it won't use LUT values
                carrier = pulse['Carrier']
                if carrier == 0:
                    continue

                if p_freq in reference_times:
                    ref_start = reference_times[p_freq]
                else:
                    reference_times[p_freq] = abs_time
                    ref_start = abs_time

                if carrier == 1:
                    # If we have a free slot for carrier 1, save it and move on
                    if latest_fp1 is None:
                        # Calculate the phase difference between this pulse and the start of the carrier change
                        carr_change_ps = ((abs_time - latest_change) * p_freq * 2)
                        # Phase sync to the reference point, then subtract the potential "double generator" offset
                        ph = utils.phase_sync(p_freq, p_phase, abs_time - ref_start) - carr_change_ps
                        for pul in vips.pulse_definitions:
                            if pul['ID'] == pulse['ID']:
                                pul['Phase'][i] = ph
                                break
                        latest_fp1 = (p_freq, ph)
                        continue
                    # If this pulse uses the same values as the last saved ones, we don't need a swap
                    elif latest_fp1 == (p_freq, p_phase):
                        continue
                    # New freq/phase, we need a reset
                    else:
                        # If we never found any pulses for carrier 2 before we needed a swap, use dummy values
                        if latest_fp2 is None:
                            latest_fp2 = (0, 0)
                        carrier_changes.append((latest_change,
                                                (latest_fp1[0], latest_fp1[1]),
                                                (latest_fp2[0], latest_fp2[1])))
                        # Save the time that we will need access to these new freq/phase values.
                        latest_change = utils.get_absolute_time(vips, time[0], time[1], i)

                        # Update global pulse definition with new phase synced value
                        ph = utils.phase_sync(p_freq, p_phase, abs_time - ref_start)
                        for pul in vips.pulse_definitions:
                            if pul['ID'] == pulse['ID']:
                                pul['Phase'][i] = ph
                                break
                        latest_fp1 = (p_freq, ph)

                        # Carrier 2 now has a free slot
                        latest_fp2 = None
                elif carrier == 2:  # Same for carrier 2
                    if latest_fp2 is None:
                        # Calculate the phase difference between this pulse and the start of the carrier change
                        carr_change_ps = ((abs_time - latest_change) * p_freq * 2)
                        # Phase sync to the reference point, then subtract the potential "double generator" offset
                        ph = utils.phase_sync(p_freq, p_phase, abs_time - ref_start) - carr_change_ps
                        for pul in vips.pulse_definitions:
                            if pul['ID'] == pulse['ID']:
                                pul['Phase'][i] = ph
                                break
                        latest_fp2 = (p_freq, ph)
                        continue
                    elif latest_fp2 == (p_freq, p_phase):
                        continue
                    else:
                        if latest_fp1 is None:
                            latest_fp1 = (0, 0)
                        carrier_changes.append((latest_change,
                                                (latest_fp1[0], latest_fp1[1]),
                                                (latest_fp2[0], latest_fp2[1])))
                        latest_change = utils.get_absolute_time(vips, time[0], time[1], i)

                        ph = utils.phase_sync(p_freq, p_phase, abs_time - ref_start)
                        for pul in vips.pulse_definitions:
                            if pul['ID'] == pulse['ID']:
                                pul['Phase'][i] = ph
                                break
                        latest_fp2 = (p_freq, ph)

                        latest_fp1 = None

        # Lock in the final freq/phase values if there are any left
        if latest_fp1 is not None or latest_fp2 is not None:
            if latest_fp2 is None:
                latest_fp2 = (0, 0)
            if latest_fp1 is None:
                latest_fp1 = (0, 0)
            carrier_changes.append((latest_change,
                                    (latest_fp1[0], latest_fp1[1]),
                                    (latest_fp2[0], latest_fp2[1])))
        port_carrier_changes[port_idx] = carrier_changes
        # Extract unique fp pairs from the change list into a LUT
        for (_, fp1, fp2) in carrier_changes:
            # If this value has not already been recorded in the LUT, add it.
            if not any([utils.are_fp_pairs_close((fp1, fp2), fpfp) for fpfp in freq_phase_values[port_idx]]):
                freq_phase_values[port_idx].append((fp1, fp2))

    return amp_values, freq_phase_values, port_carrier_changes


def apply_LUTs(vips, q):
    """
    Set up amplitude and frequency/phase LUTs on the board.
    """
    for p in range(8):
        port = p + 1
        for c in range(2):
            # Break apart our freq-phase pairs
            freq_values = []
            phase_values = []
            for (fp1, fp2) in vips.fp_matrix[p]:
                if c == 0:
                    freq = fp1[0]
                    phase = fp1[1]
                else:
                    freq = fp2[0]
                    phase = fp2[1]

                freq_values.append(freq)
                # Make sure phase stays in range
                phase_sign = np.sign(phase)
                phase = (abs(phase) % 2) * phase_sign
                assert -2 <= phase <= 2, 'Phase somehow ended up outside the range [-2,2]!'
                phase_values.append(phase * np.pi)
            # Feed our values into the tables
            if len(freq_values) > 0 and len(phase_values) > 0:
                vips.lgr.add_line(f'q.setup_freq_lut(port={port}, carrier={c + 1}, freq={freq_values}, phase={phase_values})')
                try:
                    q.setup_freq_lut(port, c+1, freq_values, phase_values)
                except ValueError as err:
                    err_str = err.args[0]
                    if err_str == 'Invalid frequency':
                        raise ValueError(f'The frequency on port {port} is outside '
                                         f'the valid range [0, 2E9] at some point!')
                    if err_str.startswith('frequency_lut can contain at most'):
                        max_num = [int(s) for s in err_str.split() if s.isdigit()][0]
                        raise ValueError(f'There are more than the max number ({max_num}) of '
                                         f'frequency/phase values on port {port}!')
                    raise err
        if len(vips.amp_matrix[p]) > 0:
            vips.lgr.add_line(f'q.setup_scale_lut(port={port}, amp={vips.amp_matrix[p]})')
            try:
                q.setup_scale_lut(port, vips.amp_matrix[p])
            except ValueError as err:
                err_str = err.args[0]
                if err_str.startswith('scale can contain at most'):
                    max_num = [int(s) for s in err_str.split() if s.isdigit()][0]
                    raise ValueError(f'There are more than the max number ({max_num}) of '
                                     f'amplitude scale values on port {port}!')
                if err_str.startswith('scale must be in'):
                    raise ValueError(f'The amplitude scale on port {port} '
                                     f'is outside the range [0, 1] at some point!')
                raise err
