import numpy as np

def generate_16qam_symbols(num_symbols):
    """Generates a stream of 16-QAM symbols."""
    # Real and Imaginary parts can be -3, -1, 1, 3
    real_part = np.random.choice([-3, -1, 1, 3], size=num_symbols)
    imag_part = np.random.choice([-3, -1, 1, 3], size=num_symbols)
    symbols = real_part + 1j * imag_part
    # Normalize symbol energy to 1
    avg_power = np.mean(np.abs(symbols) ** 2)
    symbols = symbols / np.sqrt(avg_power)
    return symbols

def apply_hammerstein_channel(signal, memory_length=5, alpha=0.8):
    """
    Applies a non-linear Hammerstein channel model to the input signal.
    Args:
        signal: Input complex signal.
        memory_length: Length of the channel impulse response.
        alpha: Non-linearity factor for tanh.
    Returns:
        distorted_signal: Output after non-linear channel.
    """
    # 1. Memoryless Non-Linearity (e.g., power amplifier saturation)
    non_linear_output = np.tanh(alpha * signal)

    # 2. Linear Dispersion (Multipath effect)
    # Create a random channel impulse response
    h = np.random.randn(memory_length) + 1j * np.random.randn(memory_length)
    h = h / np.linalg.norm(h)  # Normalize channel power

    # Convolve with the non-linear output
    distorted_signal = np.convolve(non_linear_output, h, mode='same')
    return distorted_signal

def add_awgn(signal, snr_db):
    """Adds Additive White Gaussian Noise to the signal for a given SNR (in dB)."""
    # Calculate signal power and convert SNR from dB to linear
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    # Calculate required noise power
    noise_power = sig_power / snr_linear
    # Generate complex noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def generate_16qam_data(num_symbols, snr_db):
    """Main function to generate a full dataset."""
    # Generate clean symbols
    symbols = generate_16qam_symbols(num_symbols)
    # Apply non-linear channel
    distorted_signal = apply_hammerstein_channel(symbols)
    # Add noise
    received_signal = add_awgn(distorted_signal, snr_db)
    return symbols, received_signal
