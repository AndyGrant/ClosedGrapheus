#!/bin/python3

import os
import matplotlib.pyplot as plt

DATA = [
    ('x64_mkp'           , 'L1=64'),
  # ('x64_buckets'       , 'L1=64 8x Buckets'),
    ('x64_8bit_buckets'  , 'L1=64 8x Buckets i8 Layerstack [ Gained ~7 elo ]'),
  # ('x64_psqt'          , 'Same as x64_8bit_buckets, +PSQT with 5-epochs 100x LR init'),
  # ('x64_6b_16x32x1'    , 'L1=64 6x Buckets i8 Layerstack 16x32x1'),
  # ('x64_pawn_ft_32'    , 'L1=64 8x Buckets... + L1=32 Pawns only FT'),
  # ('x48_pawn_ft_48'    , 'L1=48 8x Buckets... + L1=48 Pawns only FT'),
    ('x64_pawn_ft_32_v2' , 'L1=64 8x Buckets... + L1=32 Pawns only FT (restarted)'),
    ('x64_pawn_ft_8x16x1', 'L1=64 8x Buckets... + L1=32 Pawn FT 8x16x1 LayerStack'),
]

for run_type in ['', '_WDL']:

    plt.figure(figsize=(8, 6))

    for fname, description in DATA:

        if not os.path.exists('%s%s/loss.csv' % (fname, run_type)):
            continue

        with open('%s%s/loss.csv' % (fname, run_type)) as fin:
            y = [float(f.replace('"', '').split(',')[1]) for f in fin.readlines()[1:]]

        N = 15 if run_type == '' else 0
        y = y[N:min(len(y), 500)]
        x = list(range(N, N + len(y)))

        plt.plot(x, y, label=description)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)
    plt.savefig('loss%s.png' % (run_type))