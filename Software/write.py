# Writing/Encoding

import struct
import io
import numpy as np

VERTICAL_UNITS = 'ADU'
HORIZONTAL_UNITS = 'ns'
SAMPLE_INTERVAL = 1000/41.66
WAVEFORM_LENGTH = 251
NUMBER_OF_WAVEFORMS = 1000
PRETRIGGER_LENGTH = 64
TIMESTAMP_BYTES = 8

data = np.empty(
    (NUMBER_OF_WAVEFORMS, WAVEFORM_LENGTH + 1),
    dtype=np.uint32
)

metadata_buffer = io.BytesIO()

aux = VERTICAL_UNITS.encode('utf-8')
metadata_buffer.write(
    struct.pack(
        # Unsigned short, 2 bytes (2**16 = 65536)
        '<H',
        # Max. 65535
        len(aux)
    )
)
metadata_buffer.write(aux)

aux = HORIZONTAL_UNITS.encode('utf-8')
metadata_buffer.write(
    struct.pack(
        # Unsigned short, 2 bytes
        '<H',
        # Max. 65535
        len(aux)
    )
)
metadata_buffer.write(aux)

metadata_buffer.write(
    struct.pack(
        # Float, 4 bytes (6-7 decimals)
        '<f',
        SAMPLE_INTERVAL
    )
)

metadata_buffer.write(
    struct.pack(
        # Unsigned int, 4 bytes (2**32 = 4294967296)
        '<I',
        # Max. 4294967295
        WAVEFORM_LENGTH
    )
)

metadata_buffer.write(
    struct.pack(
        # Unsigned int, 4 bytes
        '<I',
        # Max. 4294967295
        NUMBER_OF_WAVEFORMS
    )
)

metadata_buffer.write(
    struct.pack(
        # Unsigned int, 4 bytes
        '<I',
        # Max. 4294967295
        PRETRIGGER_LENGTH
    )
)

metadata_buffer.write(
    struct.pack(
        # Unsigned short, 2 bytes
        '<H',
        # Max. 65535
        TIMESTAMP_BYTES
    )
)

metadata = metadata_buffer.getvalue()
metadata_length_bytes = len(metadata)

with open(
    'data.bin', 
    'wb'
    ) as file:

    file.write(
        struct.pack(
            # Unsigned short, 2 bytes (2**16 = 65536)
            '<H',
            # Max. 65535
            metadata_length_bytes
        )
    )

    file.write(metadata)
    file.write(data.tobytes())