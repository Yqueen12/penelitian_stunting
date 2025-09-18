import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("model_lstm_stunting.h5")

# Cek struktur model
model.summary()

# Cari layer terakhir
last_layer = model.layers[-1]
print("Nama layer terakhir:", last_layer.name)
weights = last_layer.get_weights()

# Dense biasanya punya [kernel, bias]
if len(weights) == 2:
    kernel, bias = weights
    print("Kernel shape:", kernel.shape)
    print("Bias shape:", bias.shape)

    # Simpan ke file
    np.savetxt("dense_kernel.txt", kernel, fmt="%.6f")
    np.savetxt("dense_bias.txt", bias, fmt="%.6f")
