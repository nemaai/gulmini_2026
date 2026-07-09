from scipy.io import loadmat

print("=" * 80)
print("Loading 1.mat")
print("=" * 80)

mat = loadmat(
    "1.mat",
    squeeze_me=True,
    struct_as_record=False,
)

print("\nMAT keys:")
print(mat.keys())

print("\nVariables")
print("-" * 60)

for k, v in mat.items():
    if k.startswith("__"):
        continue

    print(f"\nKey   : {k}")
    print(f"Type  : {type(v)}")

    if hasattr(v, "shape"):
        print(f"Shape : {v.shape}")

    print()

    try:
        print(dir(v))
    except:
        pass