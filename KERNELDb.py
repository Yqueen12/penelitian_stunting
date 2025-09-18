import tensorflow as tf
import numpy as np

# 1. Load model
model = tf.keras.models.load_model("model_lstm_stunting.h5")

# 2. Tampilkan struktur model
model.summary()

# 3. Loop setiap layer
for layer in model.layers:
    if 'lstm' in layer.name.lower():
        print(f"\n=== Layer: {layer.name} ===")
        weights = layer.get_weights()

        # LSTM biasanya punya 3 array: kernel, recurrent_kernel, bias
        kernel, recurrent_kernel, bias = weights

        print(f"Kernel shape: {kernel.shape}")
        print(kernel)
        
        print(f"Recurrent Kernel shape: {recurrent_kernel.shape}")
        print(recurrent_kernel)
        
        print(f"Bias shape: {bias.shape}")
        print(bias)

        # Simpan ke file supaya rapi
        np.savetxt(f"{layer.name}_kernel.txt", kernel, fmt="%.6f")
        np.savetxt(f"{layer.name}_recurrent_kernel.txt", recurrent_kernel, fmt="%.6f")
        np.savetxt(f"{layer.name}_bias.txt", bias, fmt="%.6f")
