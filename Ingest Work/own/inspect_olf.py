from scipy.io import loadmat
import numpy as np

mat = loadmat(
    "AD.mat",
    squeeze_me=True,
    struct_as_record=False,
)

print("Keys:")
print(mat.keys())

print("\nObjects:")
for k, v in mat.items():
    if not k.startswith("__"):
        print(k, type(v))

print("\nInspect:")
for k, v in mat.items():
    if not k.startswith("__"):
        print("\n==========", k, "==========")
        print(type(v))
        try:
            print(v.shape)
        except:
            pass

        print(v)
        break