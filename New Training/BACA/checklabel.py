import numpy as np

labels = np.load("BACA\label.npy")

print(type(labels))
print(labels.shape)
print(labels[:10])