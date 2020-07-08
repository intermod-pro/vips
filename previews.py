# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for ViPS's preview functionality.
"""

import numpy as np

from vivace import pulsed
import utils
import templates


def get_template_preview(vips, quant):
    """
    Construct a wave based on the envelope template
    indicated by the given quant, and return it.
    """
    with pulsed.Pulsed(ext_ref_clk=True, dry_run=True, address=vips.address) as q:
        template_defs = templates.get_template_defs(vips, q)
        # The template number X is the first character in the second word in "Template X: Preview"
        template_no = int(quant.name.split()[1][0])
        template_def = template_defs[template_no - 1]
        x, y = utils.template_def_to_points(template_def, 0, q)

        # Long drives can have length 0, which is returned as None
        if len(x) > 0:
            # Add a fake 0 at the end so the template looks nicer (squares get an extra 1 instead)
            x = np.append(x, x[-1] + 1 / q.sampling_freq)
            if y[-1] != 1:
                y = np.append(y, 0)
            else:
                y = np.append(y, 1)

            return quant.getTraceDict(y, x=x, t0=x[0], dt=(x[1] - x[0]))

        else:
            return None


def get_sequence_preview(vips, quant):
    """
    Construct a waveform from the pulse sequence information in the instrument
    and return a TraceDict with its information.
    """
    with pulsed.Pulsed(ext_ref_clk=True, dry_run=True, address=vips.address) as q:
        vips.setup_instrument(q)

        period = vips.getValue('Trigger period')
        preview_port = int(vips.getValue('Preview port'))
        preview_iter = int(vips.getValue('Preview iteration') - 1)
        preview_samples = vips.getValue('Preview sample windows')
        use_slice = vips.getValue('Enable preview slicing')
        slice_start = vips.getValue('Preview slice start')
        slice_end = min(vips.getValue('Preview slice end'), period)
        # Display nothing if the requested index is too high
        if preview_iter >= vips.iterations:
            return None

        # The preview will take the form of a list of points, with length determined by trigger period
        sampling_freq = int(q.sampling_freq)
        preview_points = np.zeros(int(sampling_freq * period) + 1)
        sample_pulses = []
        for pulse in vips.pulse_definitions:
            # Sample pulses will be handled separately
            if 'Sample' in pulse:
                if preview_samples and preview_port in vips.store_ports:
                    sample_pulses.append(pulse)
                else:
                    continue
            else:
                pulse_port = pulse['Port']
                if pulse_port != preview_port:
                    continue

                # Get pulse's envelope
                template_no = pulse['Template_no']
                template_def = vips.template_defs[template_no - 1]

                # If we have DRAG pulses on this port, get the modified envelopes
                if 'DRAG_idx' in pulse:
                    templ_x, _ = utils.template_def_to_points(template_def, preview_iter, q)
                    templ_y = vips.drag_templates[pulse['DRAG_idx']][1]
                else:
                    templ_x, templ_y = utils.template_def_to_points(template_def, preview_iter, q)
                if len(templ_y) == 0:
                    continue

                # Get other relevant parameters
                start_base, start_delta = pulse['Time']
                time = start_base + start_delta * preview_iter
                abs_time = utils.get_absolute_time(vips, start_base, start_delta, preview_iter)
                p_amp, p_freq, p_phase = utils.get_amp_freq_phase(pulse, preview_iter)

                # Calculate phase relative to latest carrier reset.
                reset_time = -1
                for (t, _, _) in vips.carrier_changes[pulse_port - 1]:
                    if t > abs_time:
                        break
                    reset_time = t
                p_phase = utils.phase_sync(p_freq, p_phase, abs_time - reset_time)
                # Construct the pulse
                if p_freq != 0 and pulse['Carrier'] != 0:
                    carrier = np.sin(2*np.pi * p_freq * templ_x + np.pi*p_phase)
                    templ_y = templ_y * carrier
                wave = templ_y * p_amp

                # Place it in the preview timeline
                pulse_index = int(time * sampling_freq)
                points_that_fit = len(preview_points[pulse_index:(pulse_index+len(wave))])
                preview_points[pulse_index:(pulse_index + points_that_fit)] += wave[:points_that_fit]

        # Display the sample windows
        for sample in sample_pulses:
            start_base, start_delta = sample['Time']
            start = start_base + start_delta * preview_iter
            duration = vips.getValue('Sampling - duration')
            wave = np.linspace(-0.1, -0.1, duration * sampling_freq)
            window_index = int(start * sampling_freq)
            points_that_fit = len(preview_points[window_index:(window_index+len(wave))])
            preview_points[window_index:(window_index + len(wave))] = wave[:points_that_fit]

    vips.reset_instrument()
    if use_slice:
        start_idx = int(slice_start * sampling_freq)
        end_idx = int(slice_end * sampling_freq)
        if end_idx - start_idx <= 0:
            return None

        preview_points = preview_points[start_idx:end_idx+1]
        times = np.linspace(slice_start, slice_end, len(preview_points))
    else:
        times = np.linspace(0, period, len(preview_points))

    return quant.getTraceDict(preview_points, x=times, t0=times[0], dt=(times[1] - times[0]))
