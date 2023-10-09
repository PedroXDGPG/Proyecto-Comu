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

def modSSB(x_t, f_c, fs):
    t = np.arange(0, len(x_t)/fs, 1/fs)  # Generar vector de tiempo
    
    # Modulación SSB
    i_t = x_t * np.cos(2 * np.pi * f_c * t)  # Señal en banda inferior
    q_t = -x_t * np.sin(2 * np.pi * f_c * t)  # Señal en banda superior
    
    s_t = i_t + q_t  # Sumar ambas señales
    
    return s_t

def transmisorSSB(x_t, f_c, fs):
    # x_t debe ser una lista con múltiples arrays (caso de 3 señales) o una sola (caso del tono)
    
    # Su código para el transmisor va aquí

    s_mod = modSSB(x_t[0], f_c, fs)
    s_t = s_mod  # eliminar cuando se tenga solución propuesta
    
    return s_t  # note que s_t es una única señal utilizando un único array, NO una lista


#TRANSMISOR SIMPLE
def transmisor(x_t):
    # x_t debe ser una lista con múltiples arrays (caso de 3 señales) o una sola (caso del tono)
    s_t = x_t[0]  # eliminar cuando se tenga solución propuesta

    return s_t



def canal(s_t):
    # Note que los parámetros mu (media) y sigma (desviación) del ruido blanco Gaussiano deben cambiarse según especificaciones
    mu = 0
    sigma = 0.1
    
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
     
    
import matplotlib.pyplot as plt
import numpy as np

def calculate_bandwidth(frequencies, psd, threshold=0.9):
    # Calculate cumulative sum of PSD
    cumulative_psd = np.cumsum(psd)
    
    # Normalize cumulative PSD
    cumulative_psd /= cumulative_psd[-1]
    
    # Find the index where cumulative power exceeds the threshold
    bandwidth_index = np.argmax(cumulative_psd >= threshold)
    
    # Get the corresponding frequencies
    lower_frequency = frequencies[bandwidth_index]
    upper_frequency = frequencies[-bandwidth_index]
    
    bandwidth = upper_frequency - lower_frequency
    
    return bandwidth

def plot_frequency_vs_psd(s_t_prima, samplerate_resampled):
    n = len(s_t_prima)

    # Calcula la transformada de Fourier
    fourier_result = np.fft.fft(s_t_prima)
    
    # Calcula las frecuencias correspondientes
    frequencies = np.fft.fftfreq(n, d=1/samplerate_resampled)

    # Calcula la PSD
    psd = np.abs(fourier_result)**2 / (n**2 * samplerate_resampled)

    # Plot the PSD
    plt.plot(frequencies, psd)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('PSD')
    plt.grid(True)
    plt.tight_layout()
    
    # Calculate and print the bandwidth
    bandwidth = calculate_bandwidth(frequencies, psd)
    print(f"The estimated bandwidth is {bandwidth} Hz.")
    
    plt.show()







# Inicio de ejecución
#TONO
file_path = "F:/TEC/tec/VIII Semestre/Comu/Proyecto/Etapa 1/datos_audio/tono.wav"

#AUDIO_1
#file_path = "F:/TEC/tec/VIII Semestre/Comu/Proyecto/Etapa 1/datos_audio/vowel_1.wav"
#BILATERAL
#Vowel1 BW de 363hz aprox
#Vowel2 BW de 430hz aprox
#Vowel3 BW de 1170hz aprox

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
print("Se envía una lista con " + str(len(x_t)) + " señales")





##############################################################INICIO SIMULACIÓN
###### -MUESTRA DE AUDIOS

# Llamar función de transmisor
s_t = transmisor(x_t)

# Llamar función que modela el canal
s_t_prima = canal(s_t)

# Sonido original########################################
print("Reproduciendo señal original:")
sd.play(tono_resampled, samplerate=samplerate_resampled)
sd.wait()

# Sonido con ruido
print("Reproduciendo señal con ruido:")
sd.play(s_t_prima, samplerate=samplerate_resampled)
sd.wait()
#########################################################

###MODULACION

# Llamar función de transmisor con portadora de 5 kHz
s_t = transmisorSSB(x_t, 2000, samplerate_resampled)

# Llamar función que modela el canal
s_t_prima = canal(s_t)

# Llamar función de receptor
m_t_reconstruida = receptor(s_t_prima, 1)  # ojo que es f_rf de prueba

#CANAL
# Graficar el ruido gaussiano en función del tiempo (0 a 0.007 segundos)
#Se debe de cambiar a 1 segundo si se quiere ver o escuchar normal
duration = 0.007  # Tiempo reducido

plot_gaussian_noise(0, 0.1, len(s_t_prima))#Ruido



# Graficar la señal original en función del tiempo (0 a 0.007 segundos)
plot_signal_vs_time(tono_resampled, samplerate_resampled, duration)

# Graficar la señal con ruido en función del tiempo (0 a 0.007 segundos)
plot_signal_vs_time(s_t_prima, samplerate_resampled, duration)




##########################################################################################
#VER LA MODULACION
def plot_mod(signal1, signal2, sample_rate, duration):
    num_samples = int(sample_rate * duration)
    time = np.arange(num_samples) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal1[:num_samples], label='tono_resampled', color='blue')
    plt.plot(time, signal2[:num_samples], label='s_t_prima', color='red')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Señal en función del tiempo')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_mod(tono_resampled, s_t_prima, samplerate_resampled, duration)
########################################################################################



# Graficar señal con ruido en el dominio de la frecuencia
plot_frequency_vs_psd(s_t_prima, samplerate_resampled)
##########################################################################



