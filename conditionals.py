# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A function for managing the setup of conditional pulses.
"""


def setup_conditionals(vips, q):
    all_conditionals = {}
    for pulse in vips.pulse_definitions:
        p_ti = pulse['Template_identifier']
        cond_on = p_ti.cond_on
        if cond_on == 'No':
            continue

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

        m1 = vips.template_matchings[cond1 - 1]
        if cond1_quad == 'I':
            matches = m1[2]
        else:
            matches = m1[3]

        if cond2 != 0:
            m2 = vips.template_matchings[cond2 - 1]
            if cond2_quad == 'I':
                matches = (matches, m2[2])
            else:
                matches = (matches, m2[3])

        p_template = vips.templates[p_ti]

        matches = tuple(matches)
        if matches not in all_conditionals:
            all_conditionals[matches] = ([], [])

        if cond_on.startswith('>'):
            all_conditionals[matches][0].append(p_template)
        else:
            all_conditionals[matches][1].append(p_template)

    for condition in all_conditionals:
        (gt_templates, lt_templates) = all_conditionals[condition]

        vips.lgr.add_line(f'q.setup_condition(matches={list(condition)}, true_templates={gt_templates}, '
                          f'false_templates={lt_templates})')
        try:
            q.setup_condition(list(condition), gt_templates, lt_templates)
        except RuntimeError:
            raise ValueError('Too many conditional pulses have been set up! A maximum of 8 '
                             'conditional pulses are allowed.')
