import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

n = 1000
k = 300

x = np.linspace(-9, 11, n)
y = (np.array([np.sign(1 - m) for m in x]) + 1)/2

x_original = np.linspace(0, 2, n)
y_original = (np.array([np.sign(1 - m) for m in x_original]) + 1)/2

yf = fft(y)

y_reconstructed = ifft(yf[k:])*1.393+1/2
x_reconstructed = np.linspace(0, 2, (n - k)//10)
print(len(y_reconstructed))

plt.plot(x_original, y_original, label='Original')
plt.plot(x_reconstructed, y_reconstructed[(n - k) * 9//20: (n - k) * 11//20], label='Reconstructed')
plt.legend()
plt.show()
