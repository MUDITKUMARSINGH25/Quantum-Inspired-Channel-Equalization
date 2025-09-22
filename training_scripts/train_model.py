from tqdm import tqdm
import numpy as np

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, epochs=25):
    """Trains and evaluates a given model."""
    print(f"   Training {model_name}...")
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=128,
                        validation_data=(X_test, y_test),
                        verbose=0) # Set verbose=1 to see progress, 0 for silent
    return history

def calculate_ber(true_symbols, predicted_symbols):
    """Calculates the overall Bit Error Rate."""
    # For 16-QAM, each symbol represents 4 bits
    # We need to map the complex symbols back to their bit representations
    # A simple way is to compare the symbols directly and assume a symbol error is 4 bit errors.
    # This is a simplification but valid for comparative analysis.
    symbol_error_rate = np.mean(true_symbols != predicted_symbols)
    # Approximate BER by assuming worst-case 4 bit errors per symbol error.
    # A more precise method would involve actual bit mapping.
    ber_approx = symbol_error_rate # Simplified for this code
    return ber_approx
