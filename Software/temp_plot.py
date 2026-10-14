import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

# ----- RUTA DEL ARCHIVO NPY -----
# MODIFICA ESTA RUTA A MANO a la dirección donde desees guardar los datos:
OUTPUT_NPY_FILE = "/home/dunelab/Documents/MASS/MASSIBO/datos_sensores_set15_meas2.npy"

# ----- CONFIGURAR TU PUERTO SERIAL -----
puerto = "/dev/ttyACM0"   # cámbialo según tu PC
baud = 9600

ser = serial.Serial(puerto, baud, timeout=1)
time.sleep(2)

# ----- ARRAYS PARA GUARDAR DATOS -----
sensor1_data = []
sensor2_data = []
sensor3_data = []
sensor4_data = []
time_data = []
system_time_data = []

start_time = time.time()

# ----- CONFIGURAR GRÁFICA -----
plt.style.use("ggplot")
fig, ax = plt.subplots()

line1, = ax.plot([], [], label="Sensor 1")
line2, = ax.plot([], [], label="Sensor 2")
line3, = ax.plot([], [], label="Sensor 3")
line4, = ax.plot([], [], label="Sensor 4")

ax.set_ylim(0, 1023)     # ADC 10 bits
ax.set_xlim(0, 10)       # Se irá expandiendo automáticamente
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Valor del sensor")
ax.legend()

# ----- FUNCIÓN QUE SE ACTUALIZA -----
def actualizar(frame):
    global ser

    if ser.in_waiting > 0:
        linea = ser.read_until().decode("utf-8").strip()

        if "sensor1" in linea:
            try:
                partes = linea.split()

                s1 = int(partes[1])
                s2 = int(partes[3])
                s3 = int(partes[5])
                s4 = int(partes[7])

                t = time.time() - start_time
                sys_t = time.time()

                # Guardar valores
                system_time_data.append(sys_t)
                time_data.append(t)
                sensor1_data.append(s1)
                sensor2_data.append(s2)
                sensor3_data.append(s3)
                sensor4_data.append(s4)

                # Guardar matriz en archivo .npy
                # Formato de columnas: [timestamp_sistema, sensor1, sensor2, sensor3, sensor4]
                matriz_datos = np.column_stack((system_time_data, sensor1_data, sensor2_data, sensor3_data, sensor4_data))
                np.save(OUTPUT_NPY_FILE, matriz_datos)

                # Actualizar las curvas completas
                line1.set_data(time_data, sensor1_data)
                line2.set_data(time_data, sensor2_data)
                line3.set_data(time_data, sensor3_data)
                line4.set_data(time_data, sensor4_data)

                # ----- EXPANDIR EJE X AUTOMÁTICAMENTE -----
                ax.set_xlim(0, max(time_data) + 1)

            except Exception as e:
                print("Error procesando:", linea, e)

    return line1, line2, line3, line4

# ----- INICIAR ANIMACIÓN -----
ani = FuncAnimation(fig, actualizar, interval=100)
plt.show()



