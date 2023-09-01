# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:06:24 2022

Plantilla para Proyecto. 
Curso Comunicaciones Electricas 1. 
Sistema de transmisión y recepción analógica

@author: lcabrera
"""

#importar bibliotecas utiles. De no tenerse alguna (import not found) se debe instalar, generalmente con pip
import scipy.signal as signal
from scipy.io import wavfile 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


#definicion de 3 bloques principales: TX, canal y RX

def transmisor(x_t):
    
    #x_t debe ser una lista con multiples array (caso de 3 señales) o una sola(caso del tono)
    atono = "F:/TEC/tec/VIII Semestre/Comu/Proyecto/Etapa 1/datos_audio/tono.wav"
    a = "F:/TEC/tec/VIII Semestre/Comu/Proyecto/Etapa 1/datos_audio/vowel_1.wav"
    b = "F:/TEC/tec/VIII Semestre/Comu/Proyecto/Etapa 1/datos_audio/vowel_2.wav"
    c = "F:/TEC/tec/VIII Semestre/Comu/Proyecto/Etapa 1/datos_audio/vowel_3.wav"
    #Su codigo para el transmisor va aca
    


    
    ####################################
    s_t=x_t[0] #eliminar cuando se tenga solucion propuesta
    
    return s_t #note que s_t es una unica señal utilizando un unico array, NO una lista

def canal(s_t):
    
    #Note que los parámetros mu (media) y sigma (desviacion) del ruido blanco Gaussiano deben cambiarse segun especificaciones
    mu=0;
    sigma=0.1;
    
    #Su codigo para el canal va aca. 
    noise = np.random.normal(mu, sigma, len(s_t))#ruido blanco
    s_t_prima = s_t + noise

    return s_t_prima


def receptor(s_t_prima,f_rf):
    
    # Note que f_rf es la frecuencia utilizada para la seleccionar la señal que se desea demodular
    
    #Su codigo para el receptor va aca  
       
    
    m_t_reconstruida=s_t_prima #eliminar cuando se tenga solucion propuesta
    
    #note que en el caso de multiples señales
    
    return m_t_reconstruida


def plot_signal_vs_time(signal, sample_rate):
    #print(len(signal))
    time = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Señal en función del tiempo')
    plt.grid(True)
    plt.show()


################### Inicio de ejecucion #####################

#Se da con ejemplo de tono, pasandolo por todo el sistema sin ningun cambio



file_path = "C:/Users/bmont/OneDrive/Documentos/2023 Segundo Semestre/Comu 1/datos_audio/datos_audio/tono.wav"

#leer tono desde archivo
samplerate_tono, tono = wavfile.read(file_path)


print("Sample Rate:", samplerate_tono)
print("Audio Data Shape:", tono.shape)


#oir tono rescatado. Esta funcion sirve tambien como transductor de salida 
#Note la importancia de la frecuencia de muestreo (samplerate), la cual es diferente a la frecuencia fm del tono.

#sd.play(tono, samplerate_tono)

#Sobremuestreo para evitar problemas de aliasing (de necesitarse)
#resampling_factor = 4
#samples_new = len(tono) * resampling_factor
#samplerate_resampled = samplerate * resampling_factor
#print('Cambiando frecuencia de muestreo de '+str(samplerate)+' a '+str(samplerate_resampled))
#data_resampled=signal.resample(tono, samples_new).astype(np.int16)
#new_length=data_resampled.shape[0] / samplerate_resampled
#time_resampled = np.linspace(0., new_length, data_resampled.shape[0])


#agregar el tono a la lista X_t requerida por el transmisor
x_t=[]  #solo para ejemplo, crear lista con el mismo tono 3 veces
x_t.append(tono)
x_t.append(tono)
x_t.append(tono)
print("Se envia una lista con "+str(len(x_t))+" señales")


#llamar funcion de transmisor
s_t=transmisor(x_t)

# llamar funcion que modela el canal
s_t_prima = canal(s_t)

# llamar funcion de receptor
m_t_reconstruida = receptor(s_t_prima, 1)#ojo que es f_rf de prueba

# Sonido original con el ruido
sd.play(m_t_reconstruida, samplerate_tono)


# Graficar la señal en función del tiempo con ruido
plot_signal_vs_time(m_t_reconstruida, samplerate_tono)
