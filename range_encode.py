import struct

class RangeCoder:
    def __init__(self):
        self.low = 0
        self.range = 0xFFFFFFFF
        self.bits_outstanding = 0
        self.buffer = []

    def encode(self, value, min_val, max_val):
        """Encodes a value using range coding."""
        range_size = max_val - min_val + 1
        self.range = self.range // range_size
        self.low += self.range * (value - min_val)

        # Emit bits while there are enough outstanding bits to do so
        while self.range <= 0xFFFFFF:
            self.buffer.append(self.low >> 24 & 0xFF)  # Mask the value to be in byte range
            self.low &= 0xFFFFFF
            self.range <<= 8
            self.bits_outstanding += 8

    def flush(self):
        """Flush remaining bits out."""
        while self.bits_outstanding > 0:
            self.buffer.append(self.low >> 24 & 0xFF)  # Mask the value to be in byte range
            self.low &= 0xFFFFFF
            self.bits_outstanding -= 8

    def get_buffer(self):
        """Return the encoded buffer."""
        return bytes(self.buffer)

    def decode(self, min_val, max_val):
        """Decodes a value from the buffer."""
        range_size = max_val - min_val + 1
        value = (self.low - self.low % self.range) // self.range
        self.low += self.range * (value - min_val)
        return value


# Compressing the data using range coding
def compress(data):
    coder = RangeCoder()
    min_val, max_val = -128, 127  # Range of int8_t values

    # Delta encoding for values near zero
    last_value = 0
    for value in data:
        delta = value - last_value
        coder.encode(delta, -128, 127)  # Use range coding for the delta
        last_value = value

    coder.flush()
    return coder.get_buffer()


# Decompressing the data using range coding
def decompress(buffer):
    coder = RangeCoder()
    coder.buffer = list(buffer)
    min_val, max_val = -128, 127  # Range of int8_t values

    last_value = 0
    decoded_values = []
    while coder.buffer:
        delta = coder.decode(min_val, max_val)
        value = last_value + delta
        decoded_values.append(value)
        last_value = value

    return decoded_values


# # Example data with int8_t values near zero
# data = [0, 1, -1, 2, -2, 0, 1, 0, -1, 2]
#
# # Compress the data
# compressed_data = compress(data)
# print("Compressed data:", list(compressed_data))
#
# # Decompress the data
# decompressed_data = decompress(compressed_data)
# print("Decompressed data:", decompressed_data)

