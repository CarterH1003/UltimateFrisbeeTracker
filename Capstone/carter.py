nums=[int(x) for x in open("PixelValues - Sheet1.csv").read().split()]
# ~ print(nums)

import numpy as np
import matplotlib.pyplot as plt

def get_dominant_freqs(time_series, sampling_rate):
    time_series -= np.mean(time_series)
    # Perform FFT
    fft_result = np.fft.fft(time_series)
    
    # Calculate corresponding frequencies
    frequencies = np.fft.fftfreq(len(fft_result), 1/sampling_rate)
    
    # Discard negative frequencies
    positive_frequencies = frequencies[frequencies > 0]
    
    # Find indices of the top N dominant frequencies (adjust N as needed)
    top_indices = np.argsort(np.abs(fft_result[:len(positive_frequencies)]))[::-1][:N]
    
    # Get dominant frequencies and corresponding magnitudes
    dominant_frequencies = positive_frequencies[top_indices]
    dominant_magnitudes = np.abs(fft_result[top_indices])
    
    return dominant_frequencies, dominant_magnitudes,frequencies,fft_result

# Example usage:
# Assume 'time_series' is your input time series array and 'sampling_rate' is the sampling rate
# Adjust 'N' to get the desired number of dominant frequencies
N = 5  # Number of dominant frequencies to retrieve

dominant_frequencies, dominant_magnitudes,frequencies,fft_result = get_dominant_freqs(nums, 240)

# Print or visualize the results
print("Dominant Frequencies:", dominant_frequencies)
print("Dominant Magnitudes:", dominant_magnitudes)

# Optional: Visualize the frequency spectrum
plt.plot(frequencies, np.abs(fft_result))
# ~ plt.plot(np.mgrid[:len(nums)], nums)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 5)
plt.show()
