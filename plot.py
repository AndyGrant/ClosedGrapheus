#!/bin/python3

import matplotlib.pyplot as plt

DATA = [
    # ('run1/loss.csv', 'L1=64'),
    # ('run2/loss.csv', '[WDL] L1=64'),
    # ('run3/loss.csv', 'L1=64 1-byte (clip 127.0 / 32)'),
    # ('run4/loss.csv', 'L1=32'),
    # ('run5/loss.csv', 'L1=64 1-byte (clip 127.0 / 32) + Material'),
    # ('run6/loss.csv', 'L1=64 1-byte (clip 127.0 / 32) + Material + 8x16x1'),
    # ('run7/loss.csv', '[WDL] L1=64 1-bits (clip 127.0 / 32) + Material + 8x16x1'),
    # ('run8/loss.csv', 'L1=64 1-byte ft/L1 (clip 127.0 / 32) + Material + 8x16x1'), # NEEDS TESTING
    ('run9/loss.csv', 'L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1'),
    # ('run10/loss.csv', '[WDL] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1'),
    # ('run11/loss.csv', '[WDL Big] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1'),
    # ('run12/loss.csv', 'L1=64 1-byte ft/L1 (clip 127.0 / 32) + Material + 16x16x1'),
    # ('runQAT3/loss.csv', 'L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1 + QAT'),
    # ('x48/loss.csv', 'L1=48 1-byte (clip 127.0 / 32) + Material + 16x16x1 + QAT'),
    ('x48_32-32/loss.csv', 'L1=48 32x32x1'),
    # ('x48_32-32_Q64/loss.csv', 'L1=48 32x32x1 Q64'),
    # ('x48_multiacti/loss.csv', 'L1=48 16x32x1 relu + sqcrelu'),
    # ('x48_nolasso/loss.csv', 'L1=48 32x32x1 No Lasso'),
    ('x64_mkp/loss.csv', 'L1=64 mirrored-kp'),
]

DATAWDL = [
    # ('run2/loss.csv', '[WDL] L1=64'),
    # ('run7/loss.csv', '[WDL] L1=64 1-bits (clip 127.0 / 32) + Material + 8x16x1'),
    # ('run10/loss.csv', '[WDL] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1'),
    # ('run11/loss.csv', '[WDL Big] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1 + 1/4 LR'),
    # ('run13/loss.csv', '[WDL Big] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1 + 1/40 LR'),
    # ('runQAT/loss.csv', '[WDL Big] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1 + 1/4 LR QAT'),
    # ('runQAT2/loss.csv', '[WDL Big] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1 + 1/4 LR QAT FT/L1'),
    # ('runQAT3A/loss.csv', '[WDL Big] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1 + 1/4 LR QAT FT'),
    # ('x48_WDL/loss.csv', '[WDL Big] L1=64 1-byte (clip 127.0 / 32) + Material + 16x16x1 + 1/40 LR QAT FT'),
    ('x48_32-32_WDL/loss.csv', 'L1=48 32x32x1 1/4LR'),
    # ('x48_32-32_Q64_WDL/loss.csv', 'L1=48 32x32x1 Q64 1/4LR'),
    ('x48_multiacti_WDL/loss.csv', 'L1=48 16x32x1 relu + sqcrelu'),
]

plt.figure(figsize=(8, 6))

for fname, description in DATA:

    with open(fname) as fin:
        y = [float(f.replace('"', '').split(',')[1]) for f in fin.readlines()[1:]]

    N = 10 # Skip first N
    y = y[N:min(len(y), 500)]
    x = list(range(N, N + len(y)))

    plt.plot(x, y, label=description)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss.png')