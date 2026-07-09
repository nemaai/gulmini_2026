---
license: mit
---

GENEEG dataset from the paper [On the challenges of detecting MCI using EEG in the wild](https://arxiv.org/abs/2501.17871)

The raw files contain raw EEG values (`.eeg` files) for patients, with corresponding:
1. `.art` files that denote level of artifact at a given time position. 0 -- No noise, 1 and more -- Noise.
2. `.evt` files that denote external stimulus at a given time position.

There is data for 17 channels: FP1, FP2, F3, F4, F7, F8, C3, C4, P3, P4, O1, O2, T3, T4, FZ, CZ, PZ. So the `.eeg` and `.art` files contain 17 space separated columns, while the `.evt` file contains only 1 column denoting the stimulus.

The `.pkl` files contain filtered data from patients as a dictionary of numpy arrays which have been artifact filtered (only 0 noise level), and contig length (200 here) chopped.

The dictionary has two keys "wmci" and "wctrl", representing the two classes MCI and Control for the WAVi dataset.

Each element in the dictionary is a list of patients data stored as a list of arrays.

```
-wmci
|-patient 1
||-contig 1 (a [17 x 200] array)
||-contig 2 
||...
||-contig N_1 (number of contigs for patient 1)
|-patient 2
||-contig 1 (a [17 x 200] array)
||-contig 2 
||...
||-contig N_2
|...
-wctrl
|-patient 1
|...
```

If you find this dataset useful, please cite as:
```
@article{mishra2025challenges,
  title={On the challenges of detecting MCI using EEG in the wild},
  author={Mishra, Aayush and Joffe, David and Telidevara, Sankara Surendra and Oakley, David S and Liu, Anqi},
  journal={arXiv preprint arXiv:2501.17871},
  year={2025}
}
```