from scipy.io import loadmat
import numpy as np

mat = loadmat("preproctrials01.mat", struct_as_record=False, squeeze_me=True)

data = mat["data"]

print("=" * 80)
print("Sampling Frequency")
print("=" * 80)
print(data.fsample)

print("\n" + "=" * 80)
print("Labels")
print("=" * 80)
print(type(data.label))
print(data.label)

print("\n" + "=" * 80)
print("Trial")
print("=" * 80)
print(type(data.trial))

trial = data.trial

print("Number of trials:", len(trial))

print()

print("First trial type")
print(type(trial[0]))

print()

print("First trial shape")
print(trial[0].shape)

print()

print("First trial first 5 channels")

print(trial[0][:5,:10])

print()

print("Time array shape")
print(data.time[0].shape)