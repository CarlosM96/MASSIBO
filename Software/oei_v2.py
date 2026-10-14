# OEI Library with UDP Timeout, Retries and Diagnostic Logging
# Compatible with Python 3

import socket
import struct
from logger import logger

class OEI(object):

    def __init__(self, ipaddr, timeout=1.5):
        """
        Inicializa la conexión UDP con DAPHNE.
        :param ipaddr: Dirección IP de DAPHNE (ej. '192.168.0.200')
        :param timeout: Tiempo máximo de espera en segundos para respuestas UDP (por defecto 1.5s).
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.timeout = timeout
        self.sock.settimeout(self.timeout)
        self.target = (ipaddr, 2001)
        logger.info(f"OEI_v2 inicializado hacia {self.target} con socket timeout = {self.timeout}s")

    def set_timeout(self, timeout):
        """Permite cambiar el timeout del socket en cualquier momento."""
        self.timeout = timeout
        self.sock.settimeout(self.timeout)
        logger.debug(f"OEI_v2 socket timeout actualizado a {self.timeout}s")

    def read(self, addr, num, max_retries=2):
        """
        Lectura de memoria con incremento de dirección.
        Utiliza formato little-endian estricto (<BBQ) para evitar bytes de alineación.
        """
        cmd = struct.pack("<BBQ", 0x00, num, addr)
        expected_size = 2 + (8 * num)

        for attempt in range(max_retries + 1):
            try:
                self.sock.sendto(cmd, self.target)
                d, a = self.sock.recvfrom(expected_size)
                if len(d) != expected_size:
                    if attempt < max_retries:
                        logger.warning(f"OEI read respuesta de tamaño inesperado ({len(d)}/{expected_size} bytes) en 0x{addr:08X}, reintentando...")
                        continue
                    raise ValueError(f"Error OEI read: tamaño recibido ({len(d)}) != esperado ({expected_size})")
                fmt = "<BB%dq" % num
                temp = struct.unpack(fmt, d)
                return temp
            except (socket.timeout, TimeoutError) as e:
                if attempt < max_retries:
                    logger.warning(f"OEI read timeout en 0x{addr:08X} (intento {attempt+1}/{max_retries+1}), reintentando...")
                else:
                    logger.error(f"OEI read FALLÓ tras {max_retries+1} intentos en 0x{addr:08X}")
                    raise TimeoutError(f"OEI read timeout en 0x{addr:08X} tras {max_retries+1} intentos") from e

    def write(self, addr, data):
        """Escritura de memoria con incremento de dirección."""
        cmd = struct.pack("<BBQ", 1, len(data), addr)
        for i in data:
            cmd += struct.pack("<Q", i)
        self.sock.sendto(cmd, self.target)

    def readf(self, addr, num, max_retries=2):
        """
        Lectura de FIFO sin incremento de dirección.
        Utiliza formato little-endian estricto (<BBQ) para evitar bytes de alineación.
        """
        cmd = struct.pack("<BBQ", 0x08, num, addr)
        expected_size = 2 + (8 * num)

        for attempt in range(max_retries + 1):
            try:
                self.sock.sendto(cmd, self.target)
                d, a = self.sock.recvfrom(expected_size)
                if len(d) != expected_size:
                    if attempt < max_retries:
                        logger.warning(f"OEI readf respuesta de tamaño inesperado ({len(d)}/{expected_size} bytes) en 0x{addr:08X}, reintentando...")
                        continue
                    raise ValueError(f"Error OEI readf: tamaño recibido ({len(d)}) != esperado ({expected_size})")
                fmt = "<BB%dQ" % num
                temp = struct.unpack(fmt, d)
                return temp
            except (socket.timeout, TimeoutError) as e:
                if attempt < max_retries:
                    logger.warning(f"OEI readf timeout en 0x{addr:08X} (intento {attempt+1}/{max_retries+1}), reintentando...")
                else:
                    logger.error(f"OEI readf FALLÓ tras {max_retries+1} intentos en 0x{addr:08X}")
                    raise TimeoutError(f"OEI readf timeout en 0x{addr:08X} tras {max_retries+1} intentos") from e

    def writef(self, addr, data):
        """Escritura de FIFO sin incremento de dirección."""
        cmd = struct.pack("<BBQ", 0x09, len(data), addr)
        for i in data:
            cmd += struct.pack("<Q", i)
        self.sock.sendto(cmd, self.target)

    def close(self):
        """Cierra el socket UDP de forma segura."""
        try:
            self.sock.close()
            logger.info("OEI_v2 socket cerrado correctamente.")
        except Exception:
            pass
