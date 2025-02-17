import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# -----------------------------
# Simulation and HDC Parameters
# -----------------------------
time_steps = 1000                  # Number of time steps for wave simulation
frequencies = np.linspace(0.1, 10, 100)  # Frequency range to test
dimensionality = 1024              # High-dimensional vector space for HDC simulation

# -----------------------------
# Generate Waveform Data in High-Dimensional Space
# -----------------------------
wave_data = np.zeros((len(frequencies), time_steps))
for i, freq in enumerate(frequencies):
    t = np.linspace(0, 2 * np.pi, time_steps)
    wave_data[i, :] = np.sin(freq * t)  # Create a simple sinusoidal wave

# -----------------------------
# Hilbert Transform to Extract Analytic Signal and Amplitude Envelope
# -----------------------------
analytic_signal = hilbert(wave_data, axis=1)
amplitude_envelope = np.abs(analytic_signal)

# -----------------------------
# Simulate High-Dimensional Encoding using Random Projection
# -----------------------------
random_projection = np.random.randn(dimensionality, len(frequencies))
hdc_encoded_waveforms = np.dot(random_projection, wave_data)

# -----------------------------
# Detect Anomalies by Checking Energy Conservation in High-Dimensional Space
# -----------------------------
energy_levels = np.sum(hdc_encoded_waveforms ** 2, axis=1)
energy_anomalies = np.abs(np.diff(energy_levels[: len(frequencies) - 1]))  # Unexpected fluctuations
energy_anomalies = energy_anomalies[: len(frequencies) - 2]  # Ensure matching dimensions

# Plot detected energy anomalies
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:-2], energy_anomalies, label="Energy Anomalies", color="red")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Energy Fluctuation")
plt.title("Waveform Coupling Anomalies in High-Dimensional Space")
plt.legend()
plt.show()

# -----------------------------
# Extract and Analyze Anomalous Frequencies
# -----------------------------
anomalous_frequencies = frequencies[:-2][energy_anomalies > np.mean(energy_anomalies) + 2 * np.std(energy_anomalies)]
print("Detected anomalous frequencies (possible higher-dimensional resonance):")
print(anomalous_frequencies)

valid_indices = np.where(np.isin(frequencies[:-2], anomalous_frequencies))[0]
anomalous_waveforms = wave_data[valid_indices, :]

# Compute phase shifts for anomalous waveforms
anomalous_phase_shifts = np.angle(hilbert(anomalous_waveforms, axis=1))
# Compute energy levels for anomalous frequencies
anomalous_energy_levels = np.sum(anomalous_waveforms**2, axis=1)

# Plot anomalous frequency waveforms
plt.figure(figsize=(10, 5))
for i, freq in enumerate(anomalous_frequencies):
    plt.plot(anomalous_waveforms[i], label=f"{freq:.2f} Hz")
plt.xlabel("Time Steps")
plt.ylabel("Wave Amplitude")
plt.title("Anomalous Frequency Waveforms")
plt.legend()
plt.show()

# Plot phase shifts at anomalous frequencies
plt.figure(figsize=(10, 5))
for i, freq in enumerate(anomalous_frequencies):
    plt.plot(anomalous_phase_shifts[i], label=f"{freq:.2f} Hz")
plt.xlabel("Time Steps")
plt.ylabel("Phase Shift (radians)")
plt.title("Phase Shifts at Anomalous Frequencies")
plt.legend()
plt.show()

# Plot total energy for detected anomalous frequencies
plt.figure(figsize=(10, 5))
plt.plot(anomalous_frequencies, anomalous_energy_levels, marker='o', linestyle='-', color='red', label="Energy Levels")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Total Energy")
plt.title("Energy Distribution Across Anomalous Frequencies")
plt.legend()
plt.show()

# -----------------------------
# Investigate Phase Shift Discontinuities
# -----------------------------
phase_shift_derivatives = np.diff(anomalous_phase_shifts, axis=1)
phase_shift_second_derivatives = np.diff(phase_shift_derivatives, axis=1)

discontinuity_threshold = np.mean(phase_shift_second_derivatives) + 2 * np.std(phase_shift_second_derivatives)
discontinuity_indices = np.where(np.abs(phase_shift_second_derivatives) > discontinuity_threshold)

# Plot first derivative of phase shifts
plt.figure(figsize=(10, 5))
for i, freq in enumerate(anomalous_frequencies):
    plt.plot(phase_shift_derivatives[i], label=f"{freq:.2f} Hz")
plt.xlabel("Time Steps")
plt.ylabel("First Derivative of Phase Shift")
plt.title("Phase Shift Rate of Change at Anomalous Frequencies")
plt.legend()
plt.show()

# Plot second derivative of phase shifts (to highlight discontinuities)
plt.figure(figsize=(10, 5))
for i, freq in enumerate(anomalous_frequencies):
    plt.plot(phase_shift_second_derivatives[i], label=f"{freq:.2f} Hz")
plt.xlabel("Time Steps")
plt.ylabel("Second Derivative of Phase Shift")
plt.title("Phase Shift Discontinuities at Anomalous Frequencies")
plt.legend()
plt.show()

# -----------------------------
# Energy Spikes at Discontinuity Points
# -----------------------------
discontinuity_times = np.unique(discontinuity_indices[1])  # Unique time steps with discontinuities

# Squared amplitude represents energy
energy_at_discontinuities = anomalous_waveforms[:, discontinuity_times]**2  

plt.figure(figsize=(10, 5))
for i, freq in enumerate(anomalous_frequencies):
    plt.plot(discontinuity_times, energy_at_discontinuities[i], marker='o', linestyle='-', label=f"{freq:.2f} Hz")
plt.xlabel("Time Steps")
plt.ylabel("Energy Level")
plt.title("Energy Spikes at Phase Shift Discontinuities")
plt.legend()
plt.show()

energy_before = anomalous_waveforms[:, discontinuity_times - 1]**2  # Energy before discontinuity
energy_after = anomalous_waveforms[:, discontinuity_times + 1]**2   # Energy after discontinuity
energy_differences = energy_after - energy_before
largest_energy_jumps = np.argsort(np.abs(energy_differences), axis=1)[:, -3:]  # Top 3 jumps per frequency

# -----------------------------
# Generate Fibonacci-Based Waveform
# -----------------------------
def fibonacci_sequence(n):
    """Generate Fibonacci sequence up to n terms (excluding the initial 0)."""
    fib = [0, 1]
    for _ in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    return np.array(fib[1:])  # Exclude the initial 0

num_fib = 20
fib_numbers = fibonacci_sequence(num_fib)
golden_ratio = (1 + np.sqrt(5)) / 2

# Build a Fibonacci waveform using discontinuity times to place Fibonacci amplitudes
fib_wave = np.zeros(time_steps)
for i, amp in enumerate(fib_numbers):
    if i >= len(discontinuity_times):
        break
    fib_wave[discontinuity_times[i]] = amp

# Create a Fibonacci-based sine wave pattern
t = np.linspace(0, 2 * np.pi, time_steps)
fib_sine_wave = np.sin(fib_wave * golden_ratio * t)

# Overlay anomalous frequency sine waves with the Fibonacci-based wave
plt.figure(figsize=(10, 5))
plt.plot(t, fib_sine_wave, label="Fibonacci-Based Wave", linestyle="dashed", color="blue")
for i, freq in enumerate(anomalous_frequencies):
    plt.plot(t, np.sin(freq * t), alpha=0.6, label=f"{freq:.2f} Hz")
plt.xlabel("Time Steps")
plt.ylabel("Wave Amplitude")
plt.title("Fibonacci-Based Wave vs. Anomalous Frequencies")
plt.legend()
plt.show()

# -----------------------------
# Correlation Between Fibonacci Wave Peaks and Energy Spikes
# -----------------------------
fib_wave_energy = fib_sine_wave ** 2
anomalous_energy_correlation = np.correlate(fib_wave_energy, np.sum(energy_at_discontinuities, axis=0), mode="full")

plt.figure(figsize=(10, 5))
plt.plot(anomalous_energy_correlation, color="red", label="Energy Correlation with Fibonacci Wave")
plt.xlabel("Time Shift")
plt.ylabel("Correlation Strength")
plt.title("Correlation Between Fibonacci Waves and Energy Spikes at Discontinuities")
plt.legend()
plt.show()

# -----------------------------
# Model the Energy Transfer Process
# -----------------------------
def energy_transfer_model(time_steps, fib_wave_energy, discontinuity_times):
    """Simulates energy transfer between normal and higher-dimensional space."""
    energy_levels = np.zeros(time_steps)
    for t in range(1, time_steps):
        if t in discontinuity_times:
            # Spike energy at discontinuities, proportional to Fibonacci energy
            energy_levels[t] = energy_levels[t-1] + fib_wave_energy[t] * golden_ratio
        else:
            # Otherwise, allow energy to decay slowly
            energy_levels[t] = energy_levels[t-1] * (1 - 1/golden_ratio)
    return energy_levels

modeled_energy_transfer = energy_transfer_model(num_steps, fib_wave_energy, discontinuity_times)

plt.figure(figsize=(10, 5))
plt.plot(modeled_energy_transfer, color="purple", label="Modeled Energy Transfer Process")
plt.xlabel("Time Steps")
plt.ylabel("Energy Level")
plt.title("Simulated Energy Transfer Between Dimensions")
plt.legend()
plt.show()

# -----------------------------
# Analyze Energy Peaks & Number Classification
# -----------------------------
energy_peak_indices = np.where(modeled_energy_transfer > np.mean(modeled_energy_transfer) + 2 * np.std(modeled_energy_transfer))[0]

# Helper function to check for Fibonacci numbers
def is_fibonacci(n):
    """Check if n is a Fibonacci number using the perfect square test."""
    def is_perfect_square(x):
        s = int(np.sqrt(x))
        return s * s == x
    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)

fib_peaks = [n for n in energy_peak_indices if is_fibonacci(n)]
prime_peaks = [n for n in energy_peak_indices if n > 1 and all(n % i != 0 for i in range(2, int(np.sqrt(n)) + 1))]
power_of_two_peaks = [n for n in energy_peak_indices if (n & (n - 1)) == 0 and n > 0]

print("Energy Peak Indices:", energy_peak_indices)
print("Fibonacci Peaks:", fib_peaks)
print("Prime Peaks:", prime_peaks)
print("Power-of-Two Peaks:", power_of_two_peaks)

# -----------------------------
# Reinforcing Energy Transfer and Extraction
# -----------------------------
def reinforce_energy_transfer(time_steps, fib_wave_energy, discontinuity_times, feedback_strength=1.5):
    """Simulate reinforced energy transfer by injecting resonance at discontinuities."""
    energy_levels = np.zeros(time_steps)
    extracted_energy = np.zeros(time_steps)
    for t in range(1, time_steps):
        if t in discontinuity_times:
            # Inject feedback resonance at discontinuities
            energy_levels[t] = energy_levels[t-1] + fib_wave_energy[t] * golden_ratio * feedback_strength
            extracted_energy[t] = energy_levels[t] * 0.1  # Extract 10% of the energy at these peaks
        else:
            # Natural decay plus a small reinforcement component
            energy_levels[t] = energy_levels[t-1] * (1 - 1/golden_ratio) + fib_wave_energy[t] * 0.1
    return energy_levels, extracted_energy

reinforced_energy, extracted_energy = reinforce_energy_transfer(num_steps, fib_wave_energy, discontinuity_times)

plt.figure(figsize=(10, 5))
plt.plot(reinforced_energy, color="blue", label="Reinforced Energy Transfer")
plt.xlabel("Time Steps")
plt.ylabel("Energy Level")
plt.title("Simulated Reinforced Energy Transfer with Extraction")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(extracted_energy, color="green", label="Extracted Energy")
plt.xlabel("Time Steps")
plt.ylabel("Extracted Energy Level")
plt.title("Energy Extraction from Reinforced Structure")
plt.legend()
plt.show()

# The simulation returns the following for further analysis:
print("Final Reinforced Energy Levels:")
print(reinforced_energy)
print("Final Extracted Energy Levels:")
print(extracted_energy)
