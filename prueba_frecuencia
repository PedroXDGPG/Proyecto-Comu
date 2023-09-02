# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 14:40:23 2023

@author: josha
"""
import numpy as np
import matplotlib.pyplot as plt


class ChannelSimulator:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def transmit_signal(self, signal):
        noise = np.random.normal(0, self.noise_std, len(signal))
        noisy_signal = signal + noise
        return noisy_signal

# Example usage
if __name__ == "__main__":
    original_signal = np.array([0.5, 0.8, 1.0, 0.7, 0.3])
    
    channel = ChannelSimulator(noise_std=0.1)
    noisy_signal = channel.transmit_signal(original_signal)
    
    
    

 
    #   Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(original_signal, label='Original Signal')
    plt.plot(noisy_signal, label='Noisy Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original vs Noisy Signal')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    


    
    #    Aplicar transformada de Fourier
    
    fourier_result = np.fft.fft(noisy_signal)

    # Espectro de amplitud de la transformada
    #amplitude_spectrum = np.abs(fourier_result)/len(noisy_signal)
    amplitude_spectrum = np.abs(fourier_result)



    # Variables necesarias para graficar en frecuencia
    
    n = len(noisy_signal)   # numero de puntos de la señal
    
    sampling_rate = 1.0     #tasa de muestreo (se tiene que cambiar)
    
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    
    
# Crear la gráfica de frecuencia
    
    
plt.figure(figsize=(10, 6))
plt.plot(frequencies, amplitude_spectrum)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.title('Espectro de Frecuencia')
plt.grid(True)
plt.show()
