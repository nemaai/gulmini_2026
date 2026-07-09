from pathlib import Path
import numpy as np

folder = Path("Patient59")

files = sorted(folder.glob("*.txt"))

print(f"Found {len(files)} channels\n")

signals = {}

for f in files:
    x = np.loadtxt(f)

    signals[f.stem] = x

    print(
        f"{f.stem:4s} "
        f"shape={x.shape} "
        f"dtype={x.dtype} "
        f"min={x.min():.3f} "
        f"max={x.max():.3f}"
    )

print("\n")

lengths = {k: len(v) for k, v in signals.items()}

print("Unique lengths:", set(lengths.values()))

print("\nFirst 10 samples of Fp1")

print(signals["Fp1"][:10])