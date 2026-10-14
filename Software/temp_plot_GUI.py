import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import serial
import serial.tools.list_ports
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import os

# ----- DIRECTORIO BASE PARA GUARDAR DATOS -----
BASE_DIR = "/home/dunelab/Documents/MASS/MASSIBO/"

class SensorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitor de Sensores de Temperatura")
        self.root.geometry("800x600")
        
        # Variables
        self.port_var = tk.StringVar()
        self.set_var = tk.StringVar(value="15")
        self.meas_var = tk.StringVar(value="2")
        
        self.ser = None
        self.is_running = False
        self.ani = None
        
        # Arrays para guardar datos
        self.sensor1_data = []
        self.sensor2_data = []
        self.sensor3_data = []
        self.sensor4_data = []
        self.time_data = []
        self.system_time_data = []
        self.start_time = 0
        self.output_file = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame de configuración
        config_frame = ttk.LabelFrame(self.root, text="Configuración")
        config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Puerto Serial
        ttk.Label(config_frame, text="Puerto USB:").grid(row=0, column=0, padx=5, pady=5)
        self.port_cb = ttk.Combobox(config_frame, textvariable=self.port_var)
        self.port_cb.grid(row=0, column=1, padx=5, pady=5)
        self.update_ports()
        
        ttk.Button(config_frame, text="Actualizar Puertos", command=self.update_ports).grid(row=0, column=2, padx=5, pady=5)
        
        # Set y Meas
        ttk.Label(config_frame, text="Set (X):").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.set_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Meas (Y):").grid(row=1, column=2, padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.meas_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Botones
        botones_frame = ttk.Frame(config_frame)
        botones_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        self.btn_start = ttk.Button(botones_frame, text="Arrancar", command=self.start_acquisition)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = ttk.Button(botones_frame, text="Detener", command=self.stop_acquisition, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        
        # Gráfica
        plt.style.use("ggplot")
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        self.line1, = self.ax.plot([], [], label="Sensor 1")
        self.line2, = self.ax.plot([], [], label="Sensor 2")
        self.line3, = self.ax.plot([], [], label="Sensor 3")
        self.line4, = self.ax.plot([], [], label="Sensor 4")
        
        self.ax.set_ylim(0, 1023)
        self.ax.set_xlim(0, 10)
        self.ax.set_xlabel("Tiempo (s)")
        self.ax.set_ylabel("Valor del sensor")
        self.ax.legend()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
    def update_ports(self):
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        self.port_cb['values'] = port_list
        if port_list:
            # Seleccionar el primero si está vacío
            if not self.port_var.get() or self.port_var.get() not in port_list:
                self.port_cb.current(0)
            
    def start_acquisition(self):
        port = self.port_var.get()
        set_val = self.set_var.get()
        meas_val = self.meas_var.get()
        
        if not port:
            messagebox.showerror("Error", "Seleccione un puerto USB")
            return
            
        if not set_val or not meas_val:
            messagebox.showerror("Error", "Ingrese valores para Set y Meas")
            return
            
        filename = f"datos_sensores_set{set_val}_meas{meas_val}.npy"
        self.output_file = os.path.join(BASE_DIR, filename)
        
        # Resetear datos por defecto
        self.sensor1_data = []
        self.sensor2_data = []
        self.sensor3_data = []
        self.sensor4_data = []
        self.time_data = []
        self.system_time_data = []
        self.start_time = time.time()
        
        # Verificar si el archivo ya existe y preguntar si se desea sobrescribir o añadir
        if os.path.exists(self.output_file):
            respuesta = messagebox.askyesnocancel(
                "Archivo existente", 
                f"El archivo {filename} ya existe.\n\n¿Desea SOBRESCRIBIRLO (Sí) o AÑADIR los datos al final (No)?"
            )
            
            if respuesta is None: # Cancelar
                return
            elif respuesta is False: # No, añadir (append)
                try:
                    datos_previos = np.load(self.output_file)
                    if len(datos_previos.shape) == 2 and datos_previos.shape[1] == 5:
                        self.system_time_data = datos_previos[:, 0].tolist()
                        self.sensor1_data = datos_previos[:, 1].tolist()
                        self.sensor2_data = datos_previos[:, 2].tolist()
                        self.sensor3_data = datos_previos[:, 3].tolist()
                        self.sensor4_data = datos_previos[:, 4].tolist()
                        
                        if len(self.system_time_data) > 0:
                            t0 = self.system_time_data[0]
                            self.time_data = [t - t0 for t in self.system_time_data]
                            self.start_time = t0
                    else:
                        messagebox.showwarning("Formato incompatible", "El archivo anterior no tiene el formato correcto (5 columnas). Se sobrescribirá.")
                        self.system_time_data = []
                        self.sensor1_data = []
                        self.sensor2_data = []
                        self.sensor3_data = []
                        self.sensor4_data = []
                        
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo cargar el archivo para añadir:\n{e}")
                    return
        
        try:
            self.ser = serial.Serial(port, 9600, timeout=1)
            time.sleep(2) # Esperar a que el Arduino se reinicie al abrir conexión serial
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el puerto {port}:\n{e}")
            return
        
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.line3.set_data([], [])
        self.line4.set_data([], [])
        self.ax.set_xlim(0, 10)
        
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.port_cb.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        # Iniciar animación
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)
        self.canvas.draw()
        
    def stop_acquisition(self):
        self.is_running = False
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
            
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None
            
        self.btn_start.config(state=tk.NORMAL)
        self.port_cb.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        
        if len(self.time_data) > 0:
            messagebox.showinfo("Información", f"Adquisición detenida.\nDatos guardados en:\n{self.output_file}")
        
    def update_plot(self, frame):
        if not self.is_running or not self.ser or not self.ser.is_open:
            return self.line1, self.line2, self.line3, self.line4
            
        try:
            # Leer todas las líneas disponibles en el buffer para evitar lag en la gráfica
            while self.ser.in_waiting > 0:
                linea = self.ser.read_until().decode("utf-8").strip()
                
                if "sensor1" in linea:
                    partes = linea.split()
                    
                    if len(partes) >= 8:
                        s1 = int(partes[1])
                        s2 = int(partes[3])
                        s3 = int(partes[5])
                        s4 = int(partes[7])
                        
                        t = time.time() - self.start_time
                        sys_t = time.time()
                        
                        self.system_time_data.append(sys_t)
                        self.time_data.append(t)
                        self.sensor1_data.append(s1)
                        self.sensor2_data.append(s2)
                        self.sensor3_data.append(s3)
                        self.sensor4_data.append(s4)
                        
                        # Actualizar datos de las líneas
                        self.line1.set_data(self.time_data, self.sensor1_data)
                        self.line2.set_data(self.time_data, self.sensor2_data)
                        self.line3.set_data(self.time_data, self.sensor3_data)
                        self.line4.set_data(self.time_data, self.sensor4_data)
                        
                        # Expandir eje X automáticamente
                        xmax = max(self.time_data) + 1 if self.time_data else 10
                        if xmax > 10:
                            self.ax.set_xlim(0, xmax)
                            
            # Guardar matriz en archivo .npy al final de la actualización del frame
            if len(self.time_data) > 0:
                matriz_datos = np.column_stack((self.system_time_data, self.sensor1_data, self.sensor2_data, self.sensor3_data, self.sensor4_data))
                np.save(self.output_file, matriz_datos)
                        
        except Exception as e:
            print("Error procesando:", e)
            
        return self.line1, self.line2, self.line3, self.line4

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorApp(root)
    
    # Manejar cierre de ventana correctamente
    def on_closing():
        if app.is_running:
            app.stop_acquisition()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
