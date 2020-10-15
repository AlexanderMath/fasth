import matplotlib.pyplot as plt
import sys
import numpy as np

blue    = "C0"
orange  = "C1"
green   = "C2"
red     = "C3"
purple  = "C4"

fig, ax = plt.subplots(1, 1, figsize=(5, 2.8))
xs      = np.array(range(64, 4096 + 1, 64))
data    = np.load("data.npz")['arr_0']

def plot(data, name, limit=48, color=None):
    if data.shape[0] < limit: limit = data.shape[0]
    mean = np.mean(data[:limit], 1)
    plt.plot(xs[:limit], mean, '-', label=name, color=color)
    plt.fill_between(xs[:limit], mean - np.std(data[:limit], 1), mean + np.std(data[:limit], 1), alpha=0.3, linewidth=0, color=color)

plot(data[:, 0], "FastH (ours)",            color=blue)
plot(data[:, 3], "Sequential", limit=512,   color=green)
plot(data[:, 2], "Cayley",                  color=red)
plot(data[:, 1], "Exponential",             color=purple)

plt.legend()

plt.xlabel("Size of matrix $d$")
plt.ylabel("Time in seconds ")

plt.xlim([64, 512])
plt.tight_layout()
plt.savefig("running_time.png")
plt.show()
