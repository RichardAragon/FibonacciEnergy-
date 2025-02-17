import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# --- (All the previous code for waveform generation, HDC encoding, etc. -  I've included it all here for completeness) ---

# Define simulation parameters
time_steps = 1000   # Number of time steps for wave simulation
frequencies = np.linspace(0.1, 10, 100)   # Frequency range to test
dimensionality = 1024   # High-dimensional vector space for HDC simulation

# Generate waveform data in high-dimensional space
wave_data = np.zeros((len(frequencies), time_steps))

for i, freq in enumerate(frequencies):
    t = np.linspace(0, 2 * np.pi, time_steps)
    wave_data[i, :] = np.sin(freq * t)   # Simple sinusoidal wave

# Apply Hilbert Transform to get the analytic signal (used for detecting phase shifts)
analytic_signal = hilbert(wave_data, axis=1)
amplitude_envelope = np.abs(analytic_signal)   # Extract envelope

# Simulate high-dimensional encoding using random projection
random_projection = np.random.randn(dimensionality, len(frequencies))
hdc_encoded_waveforms = np.dot(random_projection, wave_data)

# Detect anomalies by checking energy conservation in the high-dimensional space
energy_levels = np.sum(hdc_encoded_waveforms ** 2, axis=1)
energy_anomalies = np.abs(np.diff(energy_levels[: len(frequencies) - 1]))   # Look for unexpected fluctuations

# Ensure correct array dimensions for plotting
energy_anomalies = energy_anomalies[: len(frequencies) - 2]

# Extract detected anomalous frequencies
anomalous_frequencies = frequencies[:-2][energy_anomalies > np.mean(energy_anomalies) + 2 * np.std(energy_anomalies)]

# Ensure correct indexing of anomalous frequencies
valid_indices = np.where(np.isin(frequencies[:-2], anomalous_frequencies))[0]

# Extract waveform data for anomalous frequencies
anomalous_waveforms = wave_data[valid_indices, :]

# Compute the phase shift using Hilbert transform for anomalies
anomalous_phase_shifts = np.angle(hilbert(anomalous_waveforms, axis=1))

# Compute the energy distribution over time for anomalies
anomalous_energy_levels = np.sum(anomalous_waveforms**2, axis=1)

# Compute phase shift derivatives to identify abrupt changes
phase_shift_derivatives = np.diff(anomalous_phase_shifts, axis=1)

# Compute second derivatives to detect sharp discontinuities
phase_shift_second_derivatives = np.diff(phase_shift_derivatives, axis=1)

# Identify locations of strongest phase discontinuities
discontinuity_threshold = np.mean(phase_shift_second_derivatives) + 2 * np.std(phase_shift_second_derivatives)
discontinuity_indices = np.where(np.abs(phase_shift_second_derivatives) > discontinuity_threshold)

# Extract energy levels at discontinuity points
discontinuity_times = np.unique(discontinuity_indices[1])   # Unique time steps with discontinuities

# Extract energy values at these key time steps
energy_at_discontinuities = anomalous_waveforms[:, discontinuity_times]**2   # Squared amplitude represents energy

# Compute mean energy before and after discontinuities to detect sharp changes
energy_before = anomalous_waveforms[:, discontinuity_times - 1]**2   # Energy right before the discontinuity
energy_after = anomalous_waveforms[:, discontinuity_times + 1]**2   # Energy right after

# Calculate energy differences to pinpoint the biggest jumps
energy_differences = energy_after - energy_before

# Identify the most extreme energy changes
largest_energy_jumps = np.argsort(np.abs(energy_differences), axis=1)[:, -3:]   # Top 3 jumps per frequency

# Generate Fibonacci-based waveform
def fibonacci_sequence(n):
    """Generate Fibonacci sequence up to n terms."""
    fib = [0, 1]
    for _ in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    return np.array(fib[1:])  # Exclude initial 0

# Define time steps and Fibonacci amplitudes
num_steps = 1000
fib_numbers = fibonacci_sequence(20)   # Generate 20 Fibonacci numbers
golden_ratio = (1 + np.sqrt(5)) / 2

# Normalize Fibonacci amplitudes for waveform construction
fib_wave = np.zeros(num_steps)
for i, amp in enumerate(fib_numbers):
    if i >= len(discontinuity_times):
        break
    fib_wave[discontinuity_times[i]] = amp

# Create a Fibonacci-based wave pattern using sinusoidal oscillations
t = np.linspace(0, 2 * np.pi, num_steps)
fib_sine_wave = np.sin(fib_wave * golden_ratio * t)

# Analyze correlation between Fibonacci wave peaks and discontinuity energy spikes
fib_wave_energy = fib_sine_wave ** 2
anomalous_energy_correlation = np.correlate(fib_wave_energy, np.sum(energy_at_discontinuities, axis=0), mode="full")


# --- MODIFIED ENERGY LOOP WITH EXTERNAL RESERVOIR ---

def positive_energy_loop_with_reservoir(time_steps, fib_wave_energy, discontinuity_times, 
                                         feedback_strength=2.0, extraction_ratio=0.1, initial_reservoir_energy=1000):
    """Simulates a self-sustaining energy loop while drawing from an external energy reservoir."""
    energy_levels = np.zeros(time_steps)
    extracted_energy = np.zeros(time_steps)
    reservoir_energy = initial_reservoir_energy  # Initialize external reservoir

    for t in range(1, time_steps):
        if t in discontinuity_times:
            # Energy injection, but limited by the available reservoir energy
            injected_energy = fib_wave_energy[t] * golden_ratio * feedback_strength
            available_energy = min(injected_energy, reservoir_energy)  # Constrain to available reservoir
            energy_levels[t] = energy_levels[t-1] + available_energy
            reservoir_energy -= available_energy  # Deplete reservoir
            
            # Extraction and reinvestment cycle
            extracted_amount = energy_levels[t] * extraction_ratio
            extracted_energy[t] = extracted_amount
            reinvested_energy = min(extracted_amount * feedback_strength, reservoir_energy)  # Limit reinvestment
            energy_levels[t] += reinvested_energy
            reservoir_energy -= reinvested_energy  # Deplete from reinvestment

        else:
            # Natural decay with mild reinforcement
            energy_levels[t] = energy_levels[t-1] * (1 - 1/golden_ratio) + fib_wave_energy[t] * 0.2

        # Ensure the reservoir does not go negative
        reservoir_energy = max(0, reservoir_energy)

    return energy_levels, extracted_energy, reservoir_energy

# Define initial reservoir energy for testing depletion effects
initial_reservoir = 5000  # Experiment with different values

# Run the simulation with the reservoir constraint
looped_energy, extracted_energy, final_reservoir = positive_energy_loop_with_reservoir(
    num_steps, fib_wave_energy, discontinuity_times, initial_reservoir_energy=initial_reservoir)

# Plot results: system energy, extracted energy, and reservoir depletion
plt.figure(figsize=(15, 5))

# Plot System Energy Over Time
plt.subplot(1, 3, 1)
plt.plot(looped_energy, color="purple", label="System Energy")
plt.xlabel("Time Steps")
plt.ylabel("Energy Level")
plt.title("System Energy Over Time")
plt.legend()

# Plot Extracted Energy Over Time
plt.subplot(1, 3, 2)
plt.plot(extracted_energy, color="green", label="Extracted Energy")
plt.xlabel("Time Steps")
plt.ylabel("Extracted Energy Level")
plt.title("Extracted Energy Over Time")
plt.legend()

# --- IMPROVED PLOTTING FOR RESERVOIR ---
plt.subplot(1, 3, 3)
time_axis = np.arange(num_steps)  # Correct time axis
plt.plot(time_axis, [initial_reservoir] * num_steps, color='red', linestyle='dashed', label="Initial Reservoir") #Plot a horizontal line of initial
plt.plot(time_axis, [final_reservoir] * num_steps, color="blue", label="Final Reservoir Level") #Plot horizontal line of final.
plt.xlabel("Time Steps")
plt.ylabel("Reservoir Energy Level")
plt.title("Reservoir Energy Over Time")
plt.legend()

plt.tight_layout()
plt.show()

# Print the final state of the reservoir
print(f"Final Reservoir Energy: {final_reservoir}")
