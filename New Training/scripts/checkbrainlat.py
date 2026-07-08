import numpy as np

x = np.load("feature_001.npy")

print("ARRAY SHAPE:", x.shape)
print("DTYPE:", x.dtype)

if len(x.shape) >= 1:
    print("\nFIRST ELEMENT SHAPE:")
    try:
        print(x[0].shape)
    except:
        print("Cannot determine")

print("\nFIRST 5 VALUES:")
try:
    print(x.flatten()[:5])
except:
    pass

print("\nMIN:", np.min(x))
print("MAX:", np.max(x))

import numpy as np

x = np.load("feature_001.npy")

print("FULL SHAPE:", x.shape)

if len(x.shape) == 3:
    print("Window shape:", x[0].shape)

    if x[0].shape[1] in [14, 19]:
        print("Channels appear in axis=1")
    elif x[0].shape[0] in [14, 19]:
        print("Channels appear in axis=0")

    print("\nFirst window:")
    print(x[0][:3,:3])

    import numpy as np

x = np.load("feature_001.npy")

print("shape:", x.shape)

w = x[0]

print("window shape:", w.shape)

print("\nrow means")
print(np.mean(w, axis=1)[:10])

print("\ncol means")
print(np.mean(w, axis=0)[:10])

print("\nrow std")
print(np.std(w, axis=1)[:10])

print("\ncol std")
print(np.std(w, axis=0)[:10])