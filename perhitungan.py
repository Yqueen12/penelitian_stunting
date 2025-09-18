import numpy as np

# Fungsi aktivasi
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Load bobot
k2 = np.loadtxt("lstm_kernel.txt")              # (11, 256) -> 64 unit
rk2 = np.loadtxt("lstm_recurrent_kernel.txt")   # (64, 256)
b2 = np.loadtxt("lstm_bias.txt")                # (256,)

dense_kernel = np.loadtxt("dense_kernel.txt")   # (64,)
dense_bias = np.loadtxt("dense_bias.txt")       # scalar

# Data input
x1 = np.array([1,0,0,0,0,0,0,0,0,0,0], dtype=float)
x2 = np.array([1,0,1,0,0,1,1,0,0,0,0], dtype=float)

def lstm_step(x_t, h_prev, c_prev, kernel, recurrent_kernel, bias):
    z = x_t @ kernel + h_prev @ recurrent_kernel + bias
    u_i, u_f, u_g, u_o = np.split(z, 4)
    i = sigmoid(u_i)
    f = sigmoid(u_f)
    g = np.tanh(u_g)
    o = sigmoid(u_o)
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return h

# Hitung h_final untuk masing-masing contoh
h1 = lstm_step(x1, np.zeros(64), np.zeros(64), k2, rk2, b2)
h2 = lstm_step(x2, np.zeros(64), np.zeros(64), k2, rk2, b2)

# Hitung prediksi
logit1 = np.dot(h1, dense_kernel) + dense_bias
logit2 = np.dot(h2, dense_kernel) + dense_bias

prob1 = sigmoid(logit1)
prob2 = sigmoid(logit2)

print("Contoh 1:", prob1)
print("Contoh 2:", prob2)
