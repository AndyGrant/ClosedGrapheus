#!/bin/python3

import matplotlib.pyplot as plt

DATA = [
    ('run1/loss.csv', 'L1=64'),
    ('run3/loss.csv', 'L1=64 1-byte (clip 127.0 / 32)'),
    ('run4/loss.csv', 'L1=32'),
    ('run5/loss.csv', 'L1=64 1-byte (clip 127.0 / 32) + Material'),
]

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