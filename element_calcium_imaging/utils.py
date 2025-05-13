import numpy as np
from scipy.signal import find_peaks


def calculate_dff(trace: np.ndarray, bottom_percent: float = 10) -> np.ndarray:
    """
    Calculates df/f (delta F over F) for a single fluorescence trace.
    
    This function computes the baseline (F0) using the bottom percentile of values,
    then calculates (F - F0) / F0.

    Args:
        trace (np.ndarray): 1D NumPy array containing a single fluorescence trace.
        bottom_percent (float): Percentile to use as the baseline (default: 10).

    Returns:
        np.ndarray: df over f, with the same shape as the input trace.
    """
    # Calculate percentile for the trace
    pct_bottom = np.percentile(trace, bottom_percent)
    
    # Get values below percentile and calculate F0
    bottom_values = trace[trace <= pct_bottom]
    f0 = np.median(bottom_values)
    
    # Calculate dF/F
    return (trace - f0) / f0


def combine_trials(fissa_output, file="result", comp=0):
    """
    Fissa stores the traces in a splitted manner. This function combines the results.

    Args:
        fissa_output (npz object): fissa output
        file (str): file in the npz object.
        comp (int): signal component. 0 is the cell signal whereas the rest is the background signals.

    Returns:
        traces (np.array): traces for each cell [cell_id, time].
    """

    ntrials = fissa_output[file].shape[1]  # number of imaging (e.g. tiff) files

    traces = []
    for cell in fissa_output[file]:
        traces.append(
            np.concatenate([x[comp] for x in cell[: ntrials - 1]] + [cell[-1][0]])
        )

    return np.array(traces)


def calculate_zscore(trace: np.ndarray, sampling_rate: float, window_size_sec: float = 10.0) -> np.ndarray:
    """
    Calculate z-score using a sliding window to find the quietest period.
    
    This function:
    1. Replaces any zero values with the trace's minimum value
    2. Finds the window with minimum standard deviation (quietest period)
    3. Uses that window's mean and std to calculate z-scores
    4. Returns normalized values where 0 represents activity similar to quietest period
    
    Args:
        trace (np.ndarray): 1D array containing the fluorescence trace
        sampling_rate (float): Sampling rate in Hz
        window_size_sec (float): Size of sliding window in seconds (default: 10.0 seconds)
    
    Returns:
        np.ndarray: z-scores with same shape as input trace
    """
    # Calculate window size in samples
    window_size = int(window_size_sec * sampling_rate)
    
    # Replace zeros with minimum value
    trace_min = np.min(trace[trace != 0])  # Minimum of non-zero values
    trace[trace == 0] = trace_min
    
    # Calculate std for each possible window
    n_windows = len(trace) - window_size + 1
    window_stds = np.array([
        np.std(trace[i:i + window_size])
        for i in range(n_windows)
    ])
    
    # Find window with minimum std
    min_std_idx = np.argmin(window_stds)
    quietest_window = trace[min_std_idx:min_std_idx + window_size]
    
    # Calculate baseline statistics from quietest window
    baseline_mean = np.mean(quietest_window)
    baseline_std = np.std(quietest_window)
    
    # Calculate z-scores
    return (trace - baseline_mean) / baseline_std


def detect_events(trace: np.ndarray, sampling_rate: float, window_size_sec: float = 5.0, detection_factor: float = 4.0) -> np.ndarray:
    """
    Detect calcium events in a trace using RMS analysis and peak detection.
    
    This function:
    1. Uses sliding window RMS analysis to find quietest period
    2. Calculates noise levels using peak-to-valley analysis
    3. Normalizes signal by baseline
    4. Finds peaks in normalized signal
    5. Filters peaks by noise threshold
    
    Args:
        trace (np.ndarray): 1D array containing the fluorescence trace
        sampling_rate (float): Sampling rate in Hz
        window_size_sec (float): Size of sliding window in seconds (default: 5.0 seconds)
        detection_factor (float): Factor to multiply average peak-valley difference (default: 4.0)
    
    Returns:
        np.ndarray: Array of peak values at event locations (0 for non-events)
    """
    # Calculate window size in samples
    window_size = int(window_size_sec * sampling_rate)
    
    # Ensure trace is positive for RMS analysis
    trace_min = np.min(trace)
    if trace_min < 0:
        trace_for_rms = trace + abs(trace_min)
    else:
        trace_for_rms = trace.copy()
    
    # Calculate RMS for each sliding window
    n_windows = len(trace) - window_size + 1
    rms_values = np.array([
        np.sqrt(np.mean(window**2))  # RMS calculation
        for window in [trace_for_rms[i:i + window_size] for i in range(n_windows)]
    ])
    
    # Find window with minimum RMS
    min_rms_idx = np.argmin(rms_values)
    quietest_window = trace[min_rms_idx:min_rms_idx + window_size]
    
    # Calculate mean noise level (baseline)
    mean_noise_level = np.mean(quietest_window)
    
    # Find peaks and valleys in quietest window
    peaks, _ = find_peaks(quietest_window)
    valleys, _ = find_peaks(-quietest_window)  # Find valleys by inverting signal
    
    # Calculate peak-to-valley differences
    if len(peaks) > 0 and len(valleys) > 0:
        # Combine and sort peak/valley indices
        extrema_idx = np.sort(np.concatenate([peaks, valleys]))
        extrema_values = quietest_window[extrema_idx]
        
        # Calculate average peak-to-valley difference
        avg_peak_valley = np.mean(np.abs(np.diff(extrema_values)))
    else:
        # If no peaks/valleys found, use standard deviation as fallback
        avg_peak_valley = np.std(quietest_window)
    
    # Calculate maximum noise level (threshold)
    max_noise_level = mean_noise_level + (avg_peak_valley * detection_factor)
    
    # Calculate normalized signal
    if mean_noise_level > 0:
        normalized_signal = trace - mean_noise_level
        normalized_threshold = max_noise_level - mean_noise_level
    else:
        normalized_signal = trace + abs(mean_noise_level)
        normalized_threshold = max_noise_level + abs(mean_noise_level)
    
    # Find peaks in normalized signal & construct events array
    peak_indices, _ = find_peaks(normalized_signal, height=0)
    events = np.zeros_like(trace)
    for idx in peak_indices:
        events[idx] = normalized_signal[idx]
    
    # Filter out peaks below threshold
    events[events < normalized_threshold] = 0
    
    return events
