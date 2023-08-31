import numpy as np
import matplotlib.pyplot as plt
class CanalFisico:
    def _init_(self, desviacion_estandar=0.1):
        self.desviacion_estandar = desviacion_estandar

    def transmitir(self, senal):
        # Simular el ruido gaussiano
        ruido = np.random.normal(0, self.desviacion_estandar, len(senal))
        
        # Señal recibida es la suma de la señal original y el ruido
        senal_recibida = senal + ruido
        return senal_recibida

    
if __name__ == "_main_":
    canal = CanalFisico(desviacion_estandar=0.1)
    
    # Generar una señal de prueba (por ejemplo, una señal sinusoidal)
    tiempo = np.linspace(0, 1, num=100)
    frecuencia = 5
    senal_original = np.sin(2 * np.pi * frecuencia * tiempo)
    
    # Transmitir la señal a través del canal
    senal_recibida = canal.transmitir(senal_original)
    
    # Comparar señales visualmente
    plt.plot(tiempo, senal_original, label='Señal original')
    plt.plot(tiempo, senal_recibida, label='Señal recibida')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.show()
    
    # Calcular la diferencia entre las señales
    diferencia = np.abs(senal_original - senal_recibida)
    promedio_diferencia = np.mean(diferencia)
    print("Promedio de diferencia entre señales:", promedio_diferencia)