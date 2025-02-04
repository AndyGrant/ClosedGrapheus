import argparse
import lzma
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import tarfile
import tempfile

n_squares       = 64
n_piece_types   = 6
n_colours       = 2
n_features      = n_squares * n_piece_types * n_colours
n_pawn_features = n_colours * (n_squares - 16)

pawn_ft_in  = n_pawn_features
pawn_ft_out = 64
ft_in       = n_features
ft_out      = 64

l1_in       = (ft_out + pawn_ft_out)
l1_out      = 8
l2_in       = 8
l2_out      = 16
l3_in       = 16
l3_out      = 1
n_buckets   = 8

def quant_ft(f):
    return int(round(f * 64))

def quant_pawn_ft(f):
    return int(round(f * 32))

def quant_l1(f):
    return int(round(f * 32))

def quant_l2(f):
    return int(round(f * 32))

def example_compression(array):

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

def pre_process_ft_weights(arr, chunk_size=16):

    # Step 1: Chunk up the array into chunk_size portions
    n_chunks = len(arr) // chunk_size
    arr = arr[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)

    # Step 2: Compute the floor of the average for each chunk
    chunk_averages = np.floor(np.mean(arr, axis=1)).astype(int)

    # Step 3: Check if any values will go outside the range after +128 - avg operation
    adjusted_arrays = []; averages = []

    for chunk, avg in zip(arr, chunk_averages):

        adjusted_chunk = chunk - avg

        # Undo, if our values are not trivially small afterwards
        if np.any(adjusted_chunk < -64) or np.any(adjusted_chunk >= 64):
            adjusted_chunk = chunk; avg = 0

        averages.append(avg)
        adjusted_arrays.append(adjusted_chunk)

    # Step 4: Do a cheeky remapping of integers
    ft_weights = np.concatenate(adjusted_arrays)
    ft_weights = np.abs(ft_weights) * 2 + (ft_weights < 0)

    return ft_weights, np.array(averages)

def main():

    p = argparse.ArgumentParser()
    p.add_argument('--net', type=str, required=True)
    args = p.parse_args()

    with open(args.net, 'rb') as fin:
        ft_weights      = struct.unpack('%df' % (ft_in * ft_out             ), fin.read(ft_in * ft_out * 4              ))
        ft_bias         = struct.unpack('%df' % (ft_out                     ), fin.read(ft_out         * 4              ))
        pawn_ft_weights = struct.unpack('%df' % (pawn_ft_in * pawn_ft_out   ), fin.read(pawn_ft_in * pawn_ft_out * 4    ))
        pawn_ft_bias    = struct.unpack('%df' % (pawn_ft_out                ), fin.read(pawn_ft_out              * 4    ))
        l1_weights      = struct.unpack('%df' % (l1_in * l1_out             ), fin.read(l1_in * l1_out * 4              ))
        l1_bias         = struct.unpack('%df' % (l1_out                     ), fin.read(l1_out         * 4              ))
        l2_weights      = struct.unpack('%df' % (n_buckets * l2_in * l2_out ), fin.read(n_buckets * l2_in * l2_out * 4  ))
        l2_bias         = struct.unpack('%df' % (n_buckets * l2_out         ), fin.read(n_buckets * l2_out         * 4  ))
        l3_weights      = struct.unpack('%df' % (n_buckets * l3_in * l3_out ), fin.read(n_buckets * l3_in * l3_out * 4  ))
        l3_bias         = struct.unpack('%df' % (n_buckets * l3_out         ), fin.read(n_buckets * l3_out         * 4  ))

    ft_weights      = [          quant_ft(f) for f in ft_weights      ]
    ft_bias         = [          quant_ft(f) for f in ft_bias         ]
    pawn_ft_weights = [     quant_pawn_ft(f) for f in pawn_ft_weights ]
    pawn_ft_bias    = [     quant_pawn_ft(f) for f in pawn_ft_bias    ]
    l1_weights      = [          quant_l1(f) for f in l1_weights      ]
    l1_bias         = [          quant_l1(f) for f in l1_bias         ]
    l2_weights      = [          quant_l2(f) for f in l2_weights      ]
    l2_bias         = [          quant_l2(f) for f in l2_bias         ]
    l3_weights      = [                   f  for f in l3_weights      ]
    l3_bias         = [32 * 32 * 32 *     f  for f in l3_bias         ]

    # Convert the list into a 768xL1 numpy array
    array = np.array(ft_weights).reshape(ft_in, ft_out)

    ranges_to_delete = [
        (736, 768),  # Friendly King, not mapped onto 32 for RHS
        (440, 448),  # Friendly Pawn 8th Rank
        (384, 392),  # Friendly Pawn 1st Rank
        (320, 384),  # Enemy King
        ( 56,  64),  # Enemy Pawn 8th Rank
        (  0,   8),  # Enemy Pawn 1st Rank
    ]

    # Delete the specified ranges
    for start, end in ranges_to_delete:
        array = np.delete(array, np.s_[start:end], axis=0)

    # plt.hist(ft_weights, bins=255, color='blue', edgecolor='black')
    plt.hist(array.T.flatten(), bins=255, color='blue', edgecolor='black')
    plt.title('Histogram Example')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('histogram.png')

    ft_weights, ft_averages = pre_process_ft_weights(array.T.flatten())

    array = np.array(pawn_ft_weights).reshape(pawn_ft_in, pawn_ft_out)
    pawn_ft_weights, pawn_ft_averages = pre_process_ft_weights(array.T.flatten())

    l1_weights = np.array(l1_weights).reshape(l1_in, l1_out).T.flatten()
    l2_weights = np.array(l2_weights).reshape(l2_in, n_buckets * l2_out).T.flatten()
    l3_weights = np.array(l3_weights).reshape(l3_in, n_buckets * l3_out).T.flatten()

    l3_weight_scale = (2**15 - 1) / max(abs(l3_weights))
    l3_weights      = (l3_weights * l3_weight_scale).astype(np.int16).flatten()


    print ('#pragma once\n')
    print ('#include <stdalign.h>\n')
    print ('#include <stdint.h>\n')
    print ('const float l3_weight_scale = %f\n;' % (l3_weight_scale))
    print ('alignas(64) const uint8_t ft_weights_i8[]       = {\n    %s\n};\n' % (','.join([str(f) for f in ft_weights       ])))
    print ('alignas(64) const int8_t  ft_weights_avg[]      = {\n    %s\n};\n' % (','.join([str(f) for f in ft_averages      ])))
    print ('alignas(64) const int16_t ft_bias[]             = {\n    %s\n};\n' % (','.join([str(f) for f in ft_bias          ])))
    print ('alignas(64) const uint8_t pawn_ft_weights_i8[]  = {\n    %s\n};\n' % (','.join([str(f) for f in pawn_ft_weights  ])))
    print ('alignas(64) const int8_t  pawn_ft_weights_avg[] = {\n    %s\n};\n' % (','.join([str(f) for f in pawn_ft_averages ])))
    print ('alignas(64) const int16_t pawn_ft_bias[]        = {\n    %s\n};\n' % (','.join([str(f) for f in pawn_ft_bias     ])))
    print ('alignas(64) const int8_t  l1_weights_i8[]       = {\n    %s\n};\n' % (','.join([str(f) for f in l1_weights       ])))
    print ('alignas(64) const int16_t l1_bias_i16[]         = {\n    %s\n};\n' % (','.join([str(f) for f in l1_bias          ])))
    print ('alignas(64) const int8_t  l2_weights_i8[]       = {\n    %s\n};\n' % (','.join([str(f) for f in l2_weights       ])))
    print ('alignas(64) const int16_t l2_bias_i16[]         = {\n    %s\n};\n' % (','.join([str(f) for f in l2_bias          ])))
    print ('alignas(64) const int16_t l3_weights_i16[]      = {\n    %s\n};\n' % (','.join([str(f) for f in l3_weights       ])))
    print ('alignas(64) const float   l3_bias[]             = {\n    %s\n};\n' % (','.join([str(f) for f in l3_bias          ])))

if __name__ == '__main__':
    main()

