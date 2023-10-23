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
from scipy.signal import resample
from scipy.signal import butter, filtfilt
# Definición de 3 bloques principales: TX, canal y RX


#---------------------- TX -------------------------
def multiplexar_fdm(s_m):
    max_len = max(len(señal) for señal in s_m)
    s_t_multiplex = np.zeros(max_len, dtype=complex)  

    for señal_modulada in s_m:
        s_t_multiplex[:len(señal_modulada)] += señal_modulada
    
    return s_t_multiplex

def modSSB(x_t, f_c, fs):
    t = np.arange(0, len(x_t)/fs, 1/fs)  # Generar vector de tiempo
    
    # Modulación SSB
    i_t = x_t * np.cos(2 * np.pi * f_c * t)  # Señal en banda inferior
    q_t = -x_t * np.sin(2 * np.pi * f_c * t)  # Señal en banda superior
    
    s_t = i_t + q_t  
    
    return s_t



def demodSSB(s_t, f_c, fs, resampling_factor=5):
    t = np.arange(0, len(s_t)/fs, 1/fs)  # Generar vector de tiempo
    
    # Generar la señal de portadora de la banda inferior
    i_t = np.cos(2 * np.pi * f_c * t)  # Señal en banda inferior
    
    # Demodulación de la banda inferior
    i_demod = s_t * i_t
    
    # Filtrar la señal demodulada
    i_demod_filtrada = filtro_pasabajo(i_demod, f_c, fs)
    
    # Realizar resampling
    i_demod_resampled = resample(i_demod_filtrada, len(i_demod_filtrada) // resampling_factor)
    
    return i_demod_resampled




def transmisorSSB(x_t, f_c, fs):
    # x_t debe ser una lista con múltiples arrays (caso de 3 señales) o una sola (caso del tono)
    señales_moduladas = []

    #Modulación respecto a cada f_c
    for i in range(len(x_t)):
        señal_modulada = modSSB(x_t[i], f_c[i], fs)
        #plot_frequency_vs_psd(señal_modulada, fs)
        señal_modulada =filtro_pasabajo(señal_modulada, f_c[i], fs)
        #plot_frequency_vs_psd(señal_modulada, fs)
        señales_moduladas.append(señal_modulada)

    # Su código para el transmisor va aquí
    s_TX = multiplexar_fdm(señales_moduladas)
    
    return s_TX  # note que s_t es una única señal utilizando un único array, NO una lista

def filtro_pasabajo(s_t_prima, fcorte, fs):

    t = np.linspace(0, 1, fs, endpoint=False)
    signal_input = s_t_prima

    # T F
    signal_fft = np.fft.fft(s_t_prima)
    frequencies = np.fft.fftfreq(len(s_t_prima), 1/fs)
    filter_mask = np.where((frequencies >= -fcorte) & (frequencies <= fcorte), 1, 0)
    filtered_signal_fft = signal_fft * filter_mask

    s_t_filtrada = np.fft.ifft(filtered_signal_fft)


    return s_t_filtrada

#TRANSMISOR SIMPLE
def transmisor(x_t):
    # x_t debe ser una lista con múltiples arrays (caso de 3 señales) o una sola (caso del tono)
    s_t = x_t[0]  # eliminar cuando se tenga solución propuesta

    return s_t

#---------------------- Canal -------------------------
def canal(s_t):
    # Note que los parámetros mu (media) y sigma (desviación) del ruido blanco Gaussiano deben cambiarse según especificaciones
    mu = 0
    sigma = 0.1
    
    # Generar ruido gaussiano
    noise = np.random.normal(mu, sigma, len(s_t))
    
    # Sumar el ruido a la señal original
    s_t_prima = s_t + noise
    
    return s_t_prima

#---------------------- RX -------------------------
def receptor(s_t_prima, f_rf):
    # Note que f_rf es la frecuencia utilizada para seleccionar la señal que se desea demodular
    
    # Su código para el receptor va aquí.
    
    m_t_reconstruida = s_t_prima  # eliminar cuando se tenga solución propuesta
    
    return m_t_reconstruida

def demultiplexar_y_amplificar(salidaTX,Frf):
    
    Fif=3000   # en 2500 o menos parece haber aliasing
   
    # Señal de tiempo
    t = np.linspace(0, 1, samplerate_resampled, endpoint=False)


    # Oscilador local
    oscilador_local_largo = np.cos(2 * np.pi * (Frf - Fif) * t)
    oscilador_local = oscilador_local_largo[:len(salidaTX)]  # Recortar para que tenga el tamaño de salidaTX

    # El mixer multiplica la señal recibida por el oscilador local
    salida_mixer = np.multiply(salidaTX, oscilador_local)

    # Aplicar filtro para solo tener la señal que se quiere
    señal_demultiplexada = filtro_pasabajo(salida_mixer, Fif, samplerate_resampled)

    # Amplificar la señal demultiplexada
    demultiplexada_amplificada = np.multiply(2, señal_demultiplexada)  # Amplificar por 2

    # Devolver la señal demultiplexada amplificada
    return demultiplexada_amplificada





#---------------------- Plots -------------------------
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
    plt.ylabel('Amplitud (dBFS)')
    plt.title('Señal en función del tiempo')
    plt.grid(True)
    plt.show()

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
    plt.ylabel('PSD (dBFS^2)/Hz)')
    plt.grid(True)
    plt.tight_layout()
    
    # Calcula BW
    bandwidth = calculate_bandwidth(frequencies, psd)
    print(f"Ancho de banda Bilateral: {bandwidth} Hz.")
    
    plt.show()


#file_path1 = "C:/Users/bmont/OneDrive/Documentos/2023 Segundo Semestre/Comu 1/datos_audio/datos_audio/vowel_1.wav"
#vowel1 = wavfile.read(file_path1)

#---------------------- Audios -------------------------
def ini_audio():
    #Ruta de archivos
    path="C:/Users/bmont/OneDrive/Documentos/2023 Segundo Semestre/Comu 1/datos_audio/datos_audio/"
    #TONO
    file_path_tono = path+"tono.wav"

    #AUDIO_1
    file_path_1 =  path+"vowel_1.wav"

    #AUDIO_2
    file_path_2 =  path+"vowel_2.wav"

    #AUDIO_3
    file_path_3 =  path+"vowel_3.wav"

    #BILATERAL
    #Vowel1 BW de 363hz aprox
    #Vowel2 BW de 430hz aprox
    #Vowel3 BW de 1170hz aprox

    #--------------------------leer tono desde archivo-------------------------------
    samplerate_tono, tono = wavfile.read(file_path_tono)

    # Sobremuestreo para evitar problemas de aliasing (de necesitarse)
    resampling_factor = 5                                            # antes era 4
    samples_new = len(tono) * resampling_factor
    samplerate_resampled_tono = samplerate_tono * resampling_factor
    print('Cambiando frecuencia de muestreo de ' + str(samplerate_tono) + ' a ' + str(samplerate_resampled_tono))
    tono_resampled = signal.resample(tono, samples_new).astype(np.int16)

    #--------------------------leer vowel_1 desde archivo-------------------------------
    samplerate_vowel_1, vowel_1 = wavfile.read(file_path_1)

    # Sobremuestreo para evitar problemas de aliasing (de necesitarse)
    resampling_factor_1 = 5                                            # antes era 4
    samples_new_1 = len(vowel_1) * resampling_factor_1
    samplerate_resampled_1 = samplerate_vowel_1 * resampling_factor_1
    print('Cambiando frecuencia de muestreo de ' + str(samplerate_vowel_1) + ' a ' + str(samplerate_resampled_1))
    vowel_1_resampled = signal.resample(vowel_1, samples_new_1).astype(np.int16)

    #--------------------------leer vowel_2 desde archivo-------------------------------
    samplerate_vowel_2, vowel_2 = wavfile.read(file_path_2)

    # Sobremuestreo para evitar problemas de aliasing (de necesitarse)
    resampling_factor_2 = 5                                           # antes era 4
    samples_new_2 = len(vowel_2) * resampling_factor_2
    samplerate_resampled_2 = samplerate_vowel_2 * resampling_factor_2
    print('Cambiando frecuencia de muestreo de ' + str(samplerate_vowel_2) + ' a ' + str(samplerate_resampled_2))
    vowel_2_resampled = signal.resample(vowel_2, samples_new_2).astype(np.int16)


    #--------------------------leer vowel_3 desde archivo-------------------------------
    samplerate_vowel_3, vowel_3 = wavfile.read(file_path_3)

    # Sobremuestreo para evitar problemas de aliasing (de necesitarse)
    resampling_factor_3 = 5                                            # antes era 4
    samples_new_3 = len(vowel_3) * resampling_factor_3
    samplerate_resampled_3 = samplerate_vowel_3 * resampling_factor_3
    print('Cambiando frecuencia de muestreo de ' + str(samplerate_vowel_3) + ' a ' + str(samplerate_resampled_3))
    vowel_3_resampled = signal.resample(vowel_3, samples_new_3).astype(np.int16)


    x_t = []  # Inicialmente, la lista está vacía

    # Pedir al usuario que ingrese las señales
    print("Las opciones para la entrada de la transmisión son las siguientes:")
    print("0. Tono")
    print("1. Vowel 1")
    print("2. Vowel 2")
    print("3. Vowel 3")
    input_str = input("Ingrese las señales (máximo 3, por ejemplo, '0' para una sola señal o '1 2 3' para tres señales separadas por espacios): ")

    seleccion = input_str.split()  # Divide la entrada en una lista de números

    for opcion in seleccion:
        opcion = int(opcion)
        print(opcion)
        if opcion == 0:
            x_t.append(tono_resampled)
        elif opcion == 1:
            x_t.append(vowel_1_resampled)
        elif opcion == 2:
            x_t.append(vowel_2_resampled)
        elif opcion == 3:
            x_t.append(vowel_3_resampled)
        else:
            print(f"La opción {opcion} no es válida y será ignorada.")

    print(f"Se han agregado las señales seleccionadas a x_t: {x_t}")

    print(f"Se han agregado las señales seleccionadas a x_t:")

    return x_t,samplerate_resampled_tono,tono_resampled


# Inicio de ejecución

#---------------------- Prueba TX -------------------------
x_t, samplerate_resampled, tono_resampled = ini_audio()



#---------------------- Graficar en PSD-------------------
f_t = [10000, 20000, 30000]  # f_t debe ser una lista de 3 frecuencias de transmision

salidaTX = transmisorSSB(x_t, f_t, samplerate_resampled)
s_t = transmisor(x_t)




# RESULTADO DE LA DEMULTIPLEXACION
#f_t = [10000, 20000, 30000]
señal_demultiplexada = demultiplexar_y_amplificar(salidaTX,30000)

plot_frequency_vs_psd(señal_demultiplexada, samplerate_resampled)


# Graficar señal del transmisor con ruido en el dominio de la frecuencia
#plot_frequency_vs_psd(s_t_prima, samplerate_resampled)

plot_frequency_vs_psd(salidaTX, samplerate_resampled)
f_c = 5000
s_t = transmisor(x_t)
# Suponiendo que x_t_demod contiene las señales demoduladas
duration= 0.007
# Graficar la señal demodulada en el tiempo




x_t_demod = demodSSB(salidaTX, f_c, samplerate_resampled)
plot_signal_vs_time(x_t_demod, samplerate_resampled, duration)

##Señal original
#plot_signal_vs_time(, samplerate_resampled, duration)


# Graficar la frecuencia vs PSD
#plot_frequency_vs_psd(i_t_demod, fs)
#plot_frequency_vs_psd(q_t_demod, fs)




#---------------------- MUESTRA DE AUDIOS -------------------------
# Llamar función de transmisor
#s_t = transmisor(x_t)

# Llamar función que modela el canal
#s_t_prima = canal(s_t)

# Sonido original########################################
#print("Reproduciendo señal original:")
#sd.play(x_t[1], samplerate=samplerate_resampled)
#sd.wait()

# Sonido con ruido
#print("Reproduciendo señal con ruido:")
#sd.play(s_t_prima, samplerate=samplerate_resampled)
#sd.wait()
#########################################################












