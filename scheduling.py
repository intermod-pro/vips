# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for scheduling the emission of pulses and carriers in Vivace.
"""

import math

import utils


def setup_sequence(vips, q):
    """
    Issue commands to the board to set up the pulse sequence defined in the instrument.
    """
    # The time at which the latest emitted pulse ended, for each port
    # Used to determine when to step in the LUTs, to ensure that all parameters are ready
    # before outputting the next pulse
    latest_output = [0] * vips.N_OUT_PORTS
    setup_carriers(vips, q)
    for iter in range(vips.iterations):
        vips.lgr.add_line(f'-- Iteration {iter} --')
        for pulse in vips.pulse_definitions:
            latest_output = setup_pulse(vips, iter, latest_output, pulse, q)
        for window in vips.sample_windows:
            setup_sample_window(vips, window, iter, q)


def setup_carriers(vips, q):
    """
    Called from setup_sequence().
    Schedule the starting of Vivace's carrier generators, with appropriately timed LUT stepping.
    """
    for p, port_changes in enumerate(vips.carrier_changes):
        for i, (t, fp1, fp2) in enumerate(port_changes):
            # Step to the pulse's parameters in the LUTs
            go_to_fp(vips, q, t, p + 1, (fp1, fp2))
            # If not at the last change, calculate how long we can output until the next change
            if i != len(port_changes) - 1:
                duration = port_changes[i+1][0] - t - 2e-9
            # The last carrier of the last iteration can run until the iteration's end
            else:
                duration = (vips.trigger_period * vips.iterations) - t - 2e-9
            vips.lgr.add_line(f"q.output_carrier(time={t}, duration={duration}, port={p + 1})")
            q.output_carrier(t, duration, p+1)


def setup_pulse(vips, iteration, latest_output, pulse, q):
    """
    Called from setup_sequence(). This function handles all the setup of a single pulse definition.
    Returns latest_output, which is the time at which this pulse ends
    """
    port = pulse['Port']
    template_no = pulse['Template_no']
    template_def = vips.template_defs[template_no - 1]
    template = vips.templates[pulse['Template_identifier']]

    # Check if template is a long drive. If so, we might need to update its duration
    if 'Base' in template_def:
        (dur_base, dur_delta) = template_def['Base'], template_def['Delta']
        duration = dur_base + dur_delta * iteration
        set_long_duration(vips, template_def, template, duration)
    else:
        duration = template_def['Duration']

    # Get pulse parameters
    start_base, start_delta = pulse['Time']
    time = utils.get_absolute_time(vips, start_base, start_delta, iteration)
    p_amp, p_freq, p_phase = utils.get_amp_freq_phase(pulse, iteration)

    # Step to the pulse's parameters in the LUTs
    go_to_amp(vips, q, latest_output[port - 1], port, p_amp)

    # Set up the actual pulse command on the board
    output_pulse(vips, time, duration, template, template_def, q)
    # Store the time at which this pulse ended
    latest_output[port - 1] = time + duration
    # Check that we haven't exceeded trigger period length
    period_end_time = utils.get_absolute_time(vips, vips.trigger_period, 0, iteration)
    if latest_output[port - 1] >= period_end_time:
        raise ValueError(f'A pulse on port {port} ends after the end '
                         f'of the trigger period in iteration {iteration + 1}!')
    return latest_output


def setup_sample_window(vips, window, iteration, q):
    """
    Called from setup_sequence(). Sets up a sample window in Vivace,
    based on the given window definition.
    """
    start_base, start_delta = window['Time']
    vips.lgr.add_line(f'q.store(time={utils.get_absolute_time(vips, start_base, start_delta, iteration)})')
    q.store(utils.get_absolute_time(vips, start_base, start_delta, iteration))


def set_long_duration(vips, template_def, template, duration):
    """
    Give a command to the board to update the given long drive template's duration.
    """
    # If we are using gaussian flanks, the duration must be shortened
    if isinstance(template, tuple):
        long_duration = duration - 2 * template_def['Flank Duration']
        vips.lgr.add_line(f'update_total_duration({long_duration})')
        template[1].update_total_duration(long_duration)
    else:
        vips.lgr.add_line(f'update_total_duration({duration})')
        template.update_total_duration(duration)


def output_pulse(vips, time, duration, template, template_def, q):
    """
    Called from setup_pulse(), this method handles the calls to the board
    for outputting carrier pulses and envelope pulses for a specific pulse
    definition.
    """
    # If we are doing a long drive with gaussian flanks, we need to output three templates
    if isinstance(template, tuple):
        flank_duration = template_def['Flank Duration']
        # Rise
        rise_template = template[0]
        vips.lgr.add_line(f'q.output_pulse(time={time}, template={[rise_template]})')
        q.output_pulse(time, [rise_template])
        # Long
        long_template = template[1]
        vips.lgr.add_line(f'q.output_pulse(time={time + flank_duration}, template={[long_template]})')
        q.output_pulse(time + flank_duration, [long_template])
        # Fall
        fall_template = template[2]
        vips.lgr.add_line(f'q.output_pulse(time={time + (duration - flank_duration)}, template={[fall_template]})')
        q.output_pulse(time + (duration - flank_duration), [fall_template])
    else:
        vips.lgr.add_line(f'q.output_pulse(time={time}, template={[template]})')
        q.output_pulse(time, [template])


def setup_template_matches(vips, q):
    """
    Tell Vivace to perform template matches.
    """
    matches_by_time = {}
    for m in vips.template_matchings:
        if m[0] not in matches_by_time:
            matches_by_time[m[0]] = []
        if m[2] is None:
            matches_by_time[m[0]].extend([m[1], m[3]])
        else:
            matches_by_time[m[0]].extend([m[1], m[2], m[3], m[4]])

    for i in range(vips.iterations):
        for start_time in matches_by_time:
            abs_time = utils.get_absolute_time(vips, start_time, 0, i)
            vips.lgr.add_line(f'q.match(at_time={abs_time}, match_defs={matches_by_time[start_time]})')
            q.match(abs_time, matches_by_time[start_time])


def go_to_amp(vips, q, time, port, amp):
    """
    Find the correct LUT index for the given amplitude value and
    tell Vivace to select it in the given port's LUT.
    """
    p = port - 1

    # Find index of desired value
    for i, a in enumerate(vips.amp_matrix[p]):
        if math.isclose(a, amp):
            vips.lgr.add_line(f"q.select_scale(time={time}, idx={i}, port={port})")
            q.select_scale(time, i, port)


def go_to_fp(vips, q, time, port, fp1fp2):
    """
    Find the correct LUT index for the given frequency/phase value tuple
    and tell Vivace to select it in the given port's LUT.
    """
    p = port - 1

    for i, fpfp in enumerate(vips.fp_matrix[p]):
        if utils.are_fp_pairs_close(fp1fp2, fpfp):
            vips.lgr.add_line(f"q.select_frequency(time={time}, idx={i}, port={port})")
            q.select_frequency(time, i, port)
