import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad

# Define constants
FREQS = [1, 4, 15]
DURATION = 1.0
FS = 44100
VOLUME = 0.5

# Define functions
def func_aperiodic(x):
    return sum(np.cos(2 * np.pi * f * x) for f in FREQS) * np.exp(-2 * np.pi * x**2)

def func_periodic(x):
    return sum(np.cos(f * x) for f in FREQS)

def fourier_transform(freq):
    def integrand(t):
        return func_aperiodic(t) * np.exp(-2j * np.pi * freq * t)
    return quad(integrand, -np.inf, np.inf)[0]

def fourier_series(x, num_iter=1):
    a_0 = (1 / np.pi) * quad(func_periodic, -np.pi, np.pi)[0]
    out = a_0 / 2
    for n in range(num_iter):
        a_n = (1 / np.pi) * quad(lambda t: func_periodic(t) * np.cos(n * t), -np.pi, np.pi)[0]
        b_n = (1 / np.pi) * quad(lambda t: func_periodic(t) * np.sin(n * t), -np.pi, np.pi)[0]
        out += a_n * np.cos(n * x) + b_n * np.sin(n * x)
    return out

def plot_func(function, span, ax, n=None):
    x = np.linspace(-span/2, span/2, FS)
    y = np.vectorize(function)(x, n) if n else np.vectorize(function)(x)
    samples = (y).astype(np.float32)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=FS, output=True)
    stream.write(VOLUME * samples)
    stream.stop_stream()
    stream.close()
    return x, y

def play_sound():
    samples = (np.sin(2 * np.pi * np.arange(FS*DURATION) * 440 / FS)).astype(np.float32)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=FS, output=True)
    stream.write(VOLUME * samples)
    stream.stop_stream()
    stream.close()

# Main program
if __name__ == "__main__":
    play_sound()

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')

    x, y = plot_func(func_periodic, 50, ax1)
    sns.lineplot(x=x, y=y, ax=ax1)

    plt.show()
    plt.pause(10)

    x, y = plot_func(fourier_series, 50, ax2, 20)
    sns.lineplot(x=x, y=y, ax=ax2)

    plt.show()