# Else we want the amplitude or phase of the fourier transform
# Determine what slice of the window the user wants to look at
use_slice = self.getValue('Enable time trace slice')
start_time = None
end_time = None
if use_slice:
    start_time = self.getValue('Time trace slice start')
    end_time = self.getValue('Time trace slice end')
    slice_width = end_time - start_time
    if slice_width <= 0:
        raise ValueError('Time trace slice width cannot be 0 or negative!')
else:
    slice_width = self.getValue('Sampling - duration')

# The readout frequency can be given in a number of ways
readout_freq_mode = self.getValue('Readout frequency mode')

values = []
fp_idx_offset = 0
for window in range(len(self.results)):
    if readout_freq_mode == 'Enter manually':
        readout_freq = self.getValue('Readout frequency - manual')
    elif readout_freq_mode == 'Copy from normal pulse':
        freq_port = int(self.getValue('Readout frequency - copy port'))
        freq_pulse = int(self.getValue('Readout frequency - copy pulse'))
        readout_freq = self.getValue(f'Port {freq_port} - def {freq_pulse} - freq')
    elif readout_freq_mode == 'Copy from sweep pulse':
        use_sweep = self.getValue('Enable sweep')
        if not use_sweep:
            raise ValueError('You cannot copy readout frequency from '
                             'the sweep pulse unless it is defined!')
        sweep_port = int(self.getValue('Sweep port'))
        sweep_parameter = self.getValue('Sweep param')
        if sweep_parameter == 'Amplitude scale':
            readout_freq = self.fp_matrix[sweep_port - 1][self.sweep_fp_idx][0]
        else:
            readout_freq = self.fp_matrix[sweep_port - 1][self.sweep_fp_idx + fp_idx_offset][0]
            if (window+1) % self.samples_per_iteration == 0:
                fp_idx_offset += 1

    else:
        raise ValueError('We missed adding a case to a switch case')
    frequency_idx = int(round(readout_freq * slice_width))
    # Get slice of the window, if enabled
    if use_slice:
        slice_start_idx = int(start_time * self.SAMPLE_FREQ)
        slice_end_idx = int(end_time * self.SAMPLE_FREQ)
        window_slice = self.results[window][output_idx][slice_start_idx:slice_end_idx]
    else:  # Otherwise, just take the whole window
        window_slice = self.results[window][output_idx]
    fourier_data = np.fft.rfft(window_slice)
    if 'Amplitude' in quant.name:
        values.append(np.abs(fourier_data[frequency_idx]))
    elif 'Phase' in quant.name:
        values.append(np.angle(fourier_data[frequency_idx]))
return quant.getTraceDict(values, x=range(1, len(self.results) + 1))