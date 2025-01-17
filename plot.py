#!/bin/python3

import matplotlib.pyplot as plt

DATA = [
    # ('run1/loss.csv', 'L1=64'),
    # ('run2/loss.csv', '[WDL] L1=64'),
    # ('run3/loss.csv', 'L1=64 1-byte (clip 127.0 / 32)'),
    # ('run4/loss.csv', 'L1=32'),
    # ('run5/loss.csv', 'L1=64 1-byte (clip 127.0 / 32) + Material'),
    ('run6/loss.csv', 'L1=64 1-byte (clip 127.0 / 32) + Material + 8x16x1'),
    # ('run7/loss.csv', '[WDL] L1=64 1-bits (clip 127.0 / 32) + Material + 8x16x1'),
    ('run8/loss.csv', 'L1=64 1-byte ft/L1 (clip 127.0 / 32) + Material + 8x16x1'), # NEEDS TESTING
    ('run9/loss.csv', 'L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1'),
    # ('run10/loss.csv', '[WDL] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1'),
    # ('run11/loss.csv', '[WDL Big] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1'),
    ('run12/loss.csv', 'L1=64 1-byte ft/L1 (clip 127.0 / 32) + Material + 16x16x1'),
]

# DATA = [
#     ('run2/loss.csv', '[WDL] L1=64'),
#     ('run7/loss.csv', '[WDL] L1=64 1-bits (clip 127.0 / 32) + Material + 8x16x1'),
#     ('run10/loss.csv', '[WDL] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1'),
#     ('run11/loss.csv', '[WDL Big] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1'),
# ]

plt.figure(figsize=(8, 6))

for fname, description in DATA:

    with open(fname) as fin:
        y = [float(f.replace('"', '').split(',')[1]) for f in fin.readlines()[1:]]

    N = 25 # Skip first N
    y = y[N:min(len(y), 500)]
    x = list(range(N, N + len(y)))

    plt.plot(x, y, label=description)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss.png')