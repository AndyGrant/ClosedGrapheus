import argparse
import struct
import matplotlib.pyplot as plt
import leb128

n_squares     = 64
n_piece_types = 6
n_colours     = 2
n_features    = n_squares * n_piece_types * n_colours

n_l0 = 64
n_l1 = 1

def quant_ft(f):
    assert -128 <= f <= 127
    return int(round(f * 32))

def quant_l1(f):
    return int(round(f * 64))

def compress_int16_array(int16_array):
    compressed = bytearray()
    for num in int16_array:
        if num < 0:
            num += 1 << 8
        compressed.extend(leb128.u.encode(num))
    return compressed

def int16_to_bytearray(int16_array):
    byte_data = bytearray()
    for num in int16_array:
        byte_data.extend(struct.pack('<h', num))
    return byte_data


def main():

    p = argparse.ArgumentParser()
    p.add_argument('--net', type=str, required=True)
    args = p.parse_args()

    with open(args.net, 'rb') as fin:
        ft_weights = struct.unpack('%df' % (n_features * n_l0), fin.read(n_features * n_l0 * 4))
        ft_bias    = struct.unpack('%df' % (n_l0             ), fin.read(n_l0              * 4))
        l1_weights = struct.unpack('%df' % (2 * n_l0 * n_l1  ), fin.read(2 * n_l0 * n_l1   * 4))
        l1_bias    = struct.unpack('%df' % (n_l1             ), fin.read(n_l1              * 4))

    ft_weights = [quant_ft(f) for f in ft_weights]
    ft_bias    = [quant_ft(f) for f in ft_bias   ]
    l1_weights = [quant_l1(f) for f in l1_weights]
    l1_bias    = [quant_l1(f) for f in l1_bias   ]

    # x = compress_int16_array(ft_weights)
    # print (len(x))
    #
    # with open('foo.txt', 'wb') as fout:
    #     fout.write(x)
    #
    # exit()

    adj = [min(500, max(-500, f)) for f in ft_weights]
    plt.hist(adj, bins=255, color='blue', edgecolor='black')
    plt.title('Histogram Example')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('histogram.png')

    print ('#pragma once\n')
    print ('#include <stdalign.h>\n')
    print ('#include <stdint.h>\n')
    print ('alignas(64) const int8_t  ft_weights[] = {\n    %s\n};\n' % (','.join([str(f) for f in ft_weights])))
    print ('alignas(64) const int8_t  ft_bias[]    = {\n    %s\n};\n' % (','.join([str(f) for f in ft_bias   ])))
    print ('alignas(64) const int16_t l1_weights[] = {\n    %s\n};\n' % (','.join([str(f) for f in l1_weights])))
    print ('alignas(64) const int16_t l1_bias[]    = {\n    %s\n};\n' % (','.join([str(f) for f in l1_bias   ])))

if __name__ == '__main__':
    main()
