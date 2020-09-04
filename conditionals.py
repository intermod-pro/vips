# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A function for managing the setup of conditional pulses.
"""


def setup_conditionals(vips, q):
    for pulse in vips.pulse_definitions:
        p_ti = pulse['Template_identifier']
        cond1 = p_ti.cond1
        cond2 = p_ti.cond2
        cond1_quad = p_ti.cond1_quad
        cond2_quad = p_ti.cond2_quad
        if cond1 > len(vips.template_matchings):
            exist_str = 'exists' if len(vips.template_matchings) == 1 else 'exist'
            raise ValueError(f'A pulse on port {pulse["Port"]} has its first output '
                             f'condition set to template matching {cond1}, but only '
                             f'{len(vips.template_matchings)} {exist_str}!')
        if cond2 > len(vips.template_matchings):
            exist_str = 'exists' if len(vips.template_matchings) == 1 else 'exist'
            raise ValueError(f'A pulse on port {pulse["Port"]} has its second output '
                             f'condition set to template matching {cond2}, but only '
                             f'{len(vips.template_matchings)} {exist_str}!')
        if cond1 == 0:
            continue

        matches = []
        m1 = vips.template_matchings[cond1 - 1]
        if cond1_quad == 'I':
            matches.append(m1[1])
        else:
            matches.append(m1[3])
        # TODO port 2 stuff

        if cond2 != 0:
            m2 = vips.template_matchings[cond2 - 1]
            if cond2_quad == 'I':
                matches.append(m2[1])
            else:
                matches.append(m2[3])
            # TODO port 2

        p_template = vips.templates[p_ti]

        vips.lgr.add_line(f'q.setup_condition(matches={matches}, templates={p_template})')
        try:
            q.setup_condition(matches, p_template)
        except RuntimeError:
            raise ValueError('Too many conditional pulses have been set up! A maximum of 8 '
                             'conditional pulses are allowed.')
