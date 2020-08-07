# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A function for managing the setup of conditional pulses.
"""


def setup_conditionals(vips, q):
    for pulse in vips.pulse_definitions:
        p_ti = pulse['Template_identifier']
        cond1 = p_ti.cond1
        cond2 = p_ti.cond2
        if cond1 > len(vips.template_matchings):
            raise ValueError(f'A pulse on port {pulse["Port"]} has its first output '
                             f'condition set to template matching {cond1}, but only '
                             f'{len(vips.template_matchings)} exist(s)!')
        if cond2 > len(vips.template_matchings):
            raise ValueError(f'A pulse on port {pulse["Port"]} has its second output '
                             f'condition set to template matching {cond2}, but only '
                             f'{len(vips.template_matchings)} exist(s)!')
        if cond1 == 0:
            continue

        matches = []
        m1 = vips.template_matchings[cond1 - 1]
        matches.append(m1[1])
        if m1[2] is not None:
            matches.append(m1[2])

        if cond2 != 0:
            m2 = vips.template_matchings[cond2 - 1]
            matches.append(m2[1])
            if m2[2] is not None:
                matches.append(m2[2])

        p_template = vips.templates[p_ti]

        vips.lgr.add_line(f'q.setup_condition(matches={matches}, templates={p_template})')
        try:
            q.setup_condition(matches, p_template)
        except RuntimeError:
            raise ValueError('Too many conditional pulses have been set up! A maximum of 8 '
                             'conditional pulses are allowed.')
