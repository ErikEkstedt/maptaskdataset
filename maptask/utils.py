import matplotlib.pyplot as plt
import numpy as np

def plot_pitch_intensity(pitch, intensity, title='Pitch & intensity'):
    if not isinstance(pitch, np.ndarray):
        pitch = pitch.numpy()
    if not isinstance(intensity, np.ndarray):
        intensity = intensity.numpy()
    fig, ax1 = plt.subplots()
    plt.title(title)
    ax1.plot(pitch, 'b')
    ax1.set_xlabel('frames')
    ax1.set_ylabel('frequency', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(intensity, 'y')
    ax2.set_ylabel('intensity', color='y')
    ax2.tick_params('frames', colors='y')
    fig.tight_layout()
    plt.show()
