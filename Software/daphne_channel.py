from oei import OEI
from logger import logger
from typing import Union

class DaphneChannel:
    WAVEFORM_LENGTH = 251#64
    PRETRIGGER_LENGTH = 64

    IDENTIFIER: str

    DATA_ADDRESS: int
    TIMESTAMP_ADDRESS: int
    FIFO_WRITE_ADDRESS: int
    THRESHOLD_ADDRESS: int

    threshold_adc_units: int
    baseline_adc_units: int

    waveform_data: list
    timestamp_data: list

    write_flag_software: bool
    write_flag_firmware: bool

    fifo_last_write_address: int
    fifo_previous_last_write_address: int

    cuentas: int
    suma_amplitudes: int
    contador_ceros: int
    contador_no_validos: int


    def __init__(self,
                 identifier: str,
                 data_address: int,
                 timestamp_address: int,
                 fifo_write_address: int,
                 threshold_address: int,
                 threshold_adc_units: int,
                 baseline_adc_units: int):
        self.IDENTIFIER = identifier

        self.DATA_ADDRESS = data_address
        self.TIMESTAMP_ADDRESS = timestamp_address
        self.FIFO_WRITE_ADDRESS = fifo_write_address
        self.THRESHOLD_ADDRESS = threshold_address

        self.threshold_adc_units = threshold_adc_units
        self.baseline_adc_units = baseline_adc_units

        self.waveform_data = []
        self.timestamp_data = []

        self.cuentas = 0
        self.suma_amplitudes = 0
        self.contador_ceros = 0
        self.contador_no_validos = 0

        self.write_flag_software = True
        self.write_flag_firmware = True

        self.fifo_last_write_address = 0
        self.fifo_previous_last_write_address = 50000

    def empty_fifos(self, thing: OEI):
        logger.debug(f"CH {self.IDENTIFIER} Emptying FIFOS")
        for i in range(256):
            self.read_waveform(thing)
            self.read_timestamp(thing)
    
    def write_threshold_value(self, thing: OEI, threshold_adc_units = None):

        if threshold_adc_units is None:
            logger.debug(f"CH {self.IDENTIFIER} Writing threshold value {self.threshold_adc_units}")
            thing.write(self.THRESHOLD_ADDRESS,[self.threshold_adc_units])
            return
        
        logger.debug(f"CH {self.IDENTIFIER} Writing threshold value {threshold_adc_units}")
        thing.write(self.THRESHOLD_ADDRESS,[threshold_adc_units])
    
    def read_timestamp(self, thing: OEI):
        return thing.readf(self.TIMESTAMP_ADDRESS, 1)[2:][0]

    def read_waveform(self, thing: OEI) -> Union[tuple, None]:
        return thing.readf(self.DATA_ADDRESS, DaphneChannel.WAVEFORM_LENGTH)[2:]

    # def filter_waveform(self, waveform: tuple):
    #     dt_inicial, dt_final = waveform[0] >> 14, waveform[-1] >> 14 
    #     waveform_correct = (dt_inicial==1 or dt_inicial==2) and (dt_final==0 or dt_final==1 or dt_final==3)
    #     return waveform_correct

