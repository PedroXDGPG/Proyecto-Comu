# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:06:24 2022

Plantilla para Proyecto. 
Curso Comunicaciones Electricas 1. 
Sistema de transmisión y recepción analógica

@author: lcabrera
"""
#            Medinila Robles Pedro Fabricio
#            Montenegro Elizondo Brayan Ignacio
#            Valverde Sotovando Joshua Ariel



import scipy.signal as signal
from scipy.io import wavfile 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Definición de 3 bloques principales: TX, canal y RX

def transmisor(x_t):
    # x_t debe ser una lista con múltiples arrays (caso de 3 señales) o una sola (caso del tono)
    
    # Su código para el transmisor va aquí
    
    s_t = x_t[0]  # eliminar cuando se tenga solución propuesta
    
    return s_t  # note que s_t es una única señal utilizando un único array, NO una lista

def canal(s_t):
    # Note que los parámetros mu (media) y sigma (desviación) del ruido blanco Gaussiano deben cambiarse según especificaciones
    mu = 0
    #sigma = 0.1
    sigma = 500.1 #Sigma de prueba para poder graficar y que sea visible el ruido
    
    # Generar ruido gaussiano
    noise = np.random.normal(mu, sigma, len(s_t))
    
    # Sumar el ruido a la señal original
    s_t_prima = s_t + noise
    
    return s_t_prima

def receptor(s_t_prima, f_rf):
    # Note que f_rf es la frecuencia utilizada para seleccionar la señal que se desea demodular
    
    # Su código para el receptor va aquí.
    
    m_t_reconstruida = s_t_prima  # eliminar cuando se tenga solución propuesta
    
    return m_t_reconstruida

def plot_gaussian_noise(mu, sigma, num_samples):
    noise = np.random.normal(mu, sigma, num_samples)
    time = np.arange(num_samples)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, noise)
    plt.xlabel('Tiempo (muestras)')
    plt.ylabel('Amplitud')
    plt.title('Ruido Gaussiano en función del tiempo')
    plt.grid(True)
    plt.show()

def plot_signal_vs_time(signal, sample_rate, duration):
    num_samples = int(sample_rate * duration)
    time = np.arange(num_samples) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal[:num_samples])
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Señal en función del tiempo')
    plt.grid(True)
    plt.show()
     
    
def plot_frequency_spectrum(s_t_prima, samplerate_resampled):
    n = len(s_t_prima)
    
    # Calcula la transformada de Fourier
    fourier_result = np.fft.fft(s_t_prima)
    
    # Calcula el espectro de amplitud
    amplitude_spectrum = np.abs(fourier_result)/n  # normalizar?
    
    # Calcula las frecuencias correspondientes
    frequencies = np.fft.fftfreq(n, d=1/samplerate_resampled)
    
    # Grafica el espectro de frecuencia
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, amplitude_spectrum)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()

# Inicio de ejecución
#TONO
file_path = "C:/Users/bmont/OneDrive/Documentos/2023 Segundo Semestre/Comu 1/datos_audio/datos_audio/tono.wav"

#AUDIO_1
#file_path = "C:/Users/bmont/OneDrive/Documentos/2023 Segundo Semestre/Comu 1/datos_audio/datos_audio/vowel_2.wav"

#leer tono desde archivo
samplerate_tono, tono = wavfile.read(file_path)

# Sobremuestreo para evitar problemas de aliasing (de necesitarse)
resampling_factor = 4
samples_new = len(tono) * resampling_factor
samplerate_resampled = samplerate_tono * resampling_factor
print('Cambiando frecuencia de muestreo de ' + str(samplerate_tono) + ' a ' + str(samplerate_resampled))
tono_resampled = signal.resample(tono, samples_new).astype(np.int16)

# Agregar el tono a la lista x_t requerida por el transmisor
x_t = []  # solo para ejemplo, crear lista con el mismo tono 3 veces
x_t.append(tono_resampled)
x_t.append(tono_resampled)
x_t.append(tono_resampled)
#print("Se envía una lista con " + str(len(x_t)) + " señales")

# Llamar función de transmisor
s_t = transmisor(x_t)

# Llamar función que modela el canal
s_t_prima = canal(s_t)

# Sonido original
print("Reproduciendo señal original:")
sd.play(tono_resampled, samplerate=samplerate_resampled)
sd.wait()

# Sonido con ruido
print("Reproduciendo señal con ruido:")
sd.play(s_t_prima, samplerate=samplerate_resampled)
sd.wait()

# Llamar función de transmisor
s_t = transmisor(x_t)

# Llamar función que modela el canal
s_t_prima = canal(s_t)

# Llamar función de receptor
m_t_reconstruida = receptor(s_t_prima, 1)  # ojo que es f_rf de prueba

# Graficar el ruido gaussiano en función del tiempo (0 a 0.007 segundos)
#Se debe de cambiar a 1 segundo si se quiere ver o esuchar normal
duration = 0.007  # Tiempo reducido

plot_gaussian_noise(0, 0.1, len(s_t_prima))

# Graficar la señal original en función del tiempo (0 a 0.007 segundos)
plot_signal_vs_time(tono_resampled, samplerate_resampled, duration)

# Graficar la señal con ruido en función del tiempo (0 a 0.007 segundos)
plot_signal_vs_time(s_t_prima, samplerate_resampled, duration)

# Graficar señal con ruido en el dominio de la frecuencia
plot_frequency_spectrum(s_t_prima, samplerate_resampled)

