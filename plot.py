#!/bin/python3

import matplotlib.pyplot as plt

DATA = [
    ('x64_mkp'          , 'L1=64'),
    ('x64_buckets'      , 'L1=64 8x Buckets'),
    ('x64_8bit_buckets' , 'L1=64 8xBuckets clamp all to i8'),
]

for run_type in ['', '_WDL']:

    plt.figure(figsize=(8, 6))

    for fname, description in DATA:

        with open('%s%s/loss.csv' % (fname, run_type)) as fin:
            y = [float(f.replace('"', '').split(',')[1]) for f in fin.readlines()[1:]]

        N = 25 if run_type == '' else 0
        y = y[N:min(len(y), 500)]
        x = list(range(N, N + len(y)))

        plt.plot(x, y, label=description)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)
    plt.savefig('loss%s.png' % (run_type))