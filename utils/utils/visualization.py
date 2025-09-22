import matplotlib.pyplot as plt
import numpy as np

def plot_constellation(original, received, target, equalized, save_path=None):
    """Plots the constellation diagram at different stages."""
    plt.figure(figsize=(16, 4))

    plt.subplot(141)
    plt.scatter(np.real(original), np.imag(original), alpha=0.7, s=10, c='blue')
    plt.title("Original Transmitted Symbols")
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(142)
    plt.scatter(np.real(received), np.imag(received), alpha=0.7, s=10, c='red')
    plt.title("Received Symbols (Distorted + Noise)")
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(143)
    # Target is the true center symbol for each sample
    plt.scatter(np.real(target), np.imag(target), alpha=0.7, s=10, c='green')
    plt.title("Target Symbols (What we want to recover)")
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(144)
    plt.scatter(np.real(equalized), np.imag(equalized), alpha=0.7, s=10, c='purple')
    plt.title("cQINN Equalized Symbols")
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ber_curves(snr_range, ber_cqinn, ber_dnn, save_path=None):
    """Plots BER vs SNR curves for different models."""
    plt.figure()
    plt.semilogy(snr_range, ber_cqinn, 'o-', linewidth=2, markersize=8, label='cQINN Equalizer')
    plt.semilogy(snr_range, ber_dnn, 's--', linewidth=2, markersize=8, label='DNN Equalizer')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER Performance Comparison')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(cqinn_history, dnn_history, save_path=None):
    """Plots the training accuracy and loss history for both models."""
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(cqinn_history.history['accuracy'], label='cQINN Training Acc')
    plt.plot(cqinn_history.history['val_accuracy'], label='cQINN Validation Acc')
    plt.plot(dnn_history.history['accuracy'], label='DNN Training Acc')
    plt.plot(dnn_history.history['val_accuracy'], label='DNN Validation Acc')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plt.plot(cqinn_history.history['loss'], label='cQINN Training Loss')
    plt.plot(cqinn_history.history['val_loss'], label='cQINN Validation Loss')
    plt.plot(dnn_history.history['loss'], label='DNN Training Loss')
    plt.plot(dnn_history.history['val_loss'], label='DNN Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()utils/visualization.pyutils/visualization.pyutils/visualization.py
