import argparse
import lzma
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import tarfile
import tempfile

n_squares     = 64
n_piece_types = 6
n_colours     = 2
n_features    = n_squares * n_piece_types * n_colours

n_l0 = 48
n_l1 = 16
n_l2 = 16
n_l3 = 1

def quant_ft(f):
    x = int(round(f * 32))
    return x
    if abs(x) <= 2:
        return 0
    return x

def quant_l1(f):
    return int(round(f * 64))

def quant_l2(f):
    return int(round(f * 64))

def quant_l3(f):
    return int(round(f * 64))




def do_the_thing(array):

    # Create a temporary file to save the numpy array
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
        # Save the numpy array to the temporary file in .npy format
        np.save(tmp_file.name, array)

        # Compress the .npy file using lzma
        lzma_name = tmp_file.name + '.xz'
        with open(tmp_file.name, 'rb') as f_in, lzma.open(lzma_name, 'wb') as f_out:
            f_out.writelines(f_in)

        # Compress the .xz file using tar.gz
        tar_name = lzma_name + '.tar.gz'
        with tarfile.open(tar_name, 'w:gz') as tar:
            tar.add(lzma_name, arcname=os.path.basename(lzma_name))

        # Get the compressed file size
        compressed_size = os.path.getsize(tar_name)

        # Optionally, clean up temporary files
        os.remove(tmp_file.name)
        os.remove(lzma_name)
        os.remove(tar_name)

    return compressed_size

def main():

    p = argparse.ArgumentParser()
    p.add_argument('--net', type=str, required=True)
    args = p.parse_args()

    with open(args.net, 'rb') as fin:
        ft_weights = struct.unpack('%df' % (n_features * n_l0), fin.read(n_features * n_l0 * 4))
        ft_bias    = struct.unpack('%df' % (n_l0             ), fin.read(n_l0              * 4))
        l1_weights = struct.unpack('%df' % (2 * n_l0 * n_l1  ), fin.read(2 * n_l0 * n_l1   * 4))
        l1_bias    = struct.unpack('%df' % (n_l1             ), fin.read(n_l1              * 4))
        l2_weights = struct.unpack('%df' % (n_l1 * n_l2      ), fin.read(n_l1 * n_l2       * 4))
        l2_bias    = struct.unpack('%df' % (n_l2             ), fin.read(n_l2              * 4))
        l3_weights = struct.unpack('%df' % (n_l2 * n_l3      ), fin.read(n_l2 * n_l3       * 4))
        l3_bias    = struct.unpack('%df' % (n_l3             ), fin.read(n_l3              * 4))

    ft_weights = [     quant_ft(f) for f in ft_weights]
    ft_bias    = [     quant_ft(f) for f in ft_bias   ]
    l1_weights = [     quant_l1(f) for f in l1_weights]
    l1_bias    = [32 * quant_l1(f) for f in l1_bias   ]
    l2_weights = [     f for f in l2_weights]
    l2_bias    = [32 * 64 * f for f in l2_bias   ]
    l3_weights = [     f for f in l3_weights]
    l3_bias    = [32 * 64 * f for f in l3_bias   ]

    # Convert the list into a 768xL1 numpy array
    array = np.array(ft_weights).reshape(n_features, n_l0)

    # Zero out the specified ranges
    array[0:8, :]     = 0  # White Pawn 1st
    array[56:64, :]   = 0  # White Pawn 8th
    array[384:392, :] = 0  # Black Pawn 1st
    array[440:448, :] = 0  # Black Pawn 8th

    # best = do_the_thing(array.T.flatten())
    # print ('Best: ' , best)

    while True:

        break

        # Shuffle the indices of the second dimension (columns)
        shuffled_indices = np.random.permutation(array.shape[1])

        # Reorder the array based on the shuffled indices
        shuffled_array = array[:, shuffled_indices]

        x = do_the_thing(shuffled_array.T.flatten())

        if x < best:
            best = x
            print ('Best: ', best)
            print (shuffled_indices)

        # if best < 39000:
        #     array = shuffled_array
        #     break

    ft_weights = array.T.flatten()
    l1_weights = np.array(l1_weights).reshape(2 * n_l0, n_l1).T.flatten()
    l2_weights = np.array(l2_weights).reshape(    n_l1, n_l2).T.flatten()

    plt.hist(ft_weights, bins=255, color='blue', edgecolor='black')

    # plt.hist(l1_weights, bins=255, color='blue', edgecolor='black')

    plt.title('Histogram Example')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('histogram.png')

    print ('#pragma once\n')
    print ('#include <stdalign.h>\n')
    print ('#include <stdint.h>\n')
    print ('alignas(64) const int8_t  ft_weights[] = {\n    %s\n};\n' % (','.join([str(f) for f in ft_weights])))
    print ('alignas(64) const int16_t ft_bias[]    = {\n    %s\n};\n' % (','.join([str(f) for f in ft_bias   ])))
    print ('alignas(64) const int16_t l1_weights[] = {\n    %s\n};\n' % (','.join([str(f) for f in l1_weights])))
    print ('alignas(64) const int32_t l1_bias[]    = {\n    %s\n};\n' % (','.join([str(f) for f in l1_bias   ])))
    print ('alignas(64) const float   l2_weights[] = {\n    %s\n};\n' % (','.join([str(f) for f in l2_weights])))
    print ('alignas(64) const float   l2_bias[]    = {\n    %s\n};\n' % (','.join([str(f) for f in l2_bias   ])))
    print ('alignas(64) const float   l3_weights[] = {\n    %s\n};\n' % (','.join([str(f) for f in l3_weights])))
    print ('alignas(64) const float   l3_bias[]    = {\n    %s\n};\n' % (','.join([str(f) for f in l3_bias   ])))

if __name__ == '__main__':
    main()

