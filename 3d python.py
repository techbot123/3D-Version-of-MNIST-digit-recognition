import sys
sys.path.append("../input")
import h5py
import os
import numpy as np
from IPython.display import Image
from voxelgrid import VoxelGrid
from matplotlib import pyplot as plt
%matplotlib inline
plt.rcParams['image.interpolation'] = None
plt.rcParams['image.cmap'] = 'gray'
Read data from HDF5 file
with h5py.File("../input/train_small.h5", "r") as hf:

    a = hf["0"]
    b = hf["1"]

    digit_a = (a["img"][:], a["points"][:], a.attrs["label"])
    digit_b = (b["img"][:], b["points"][:], b.attrs["label"])

plt.subplot(121)
plt.title("DIGIT A: " + str(digit_a[2]))
plt.imshow(digit_a[0])

plt.subplot(122)
plt.title("DIGIT B: " + str(digit_b[2]))
plt.imshow(digit_b[0])
'''
Generate VoxelGrid
VoxelGrid

The VoxelGrid will fit an axis-aligned bounding box around the point cloud and then subdivide the box in segments along each corresponding axis.

The x_y_z argument indicates how many segments we want per axis.

We will split each axis in 8 voxels wich would be equivalent to use the 3th level of an Octree.

This will generate a total of 512 different voxels.
'''
a_voxelgrid = VoxelGrid(digit_a[1], x_y_z=[8, 8, 8])
b_voxelgrid = VoxelGrid(digit_b[1], x_y_z=[8, 8, 8])

# point coordinates
digit_a[1][340]

def plot_colorfull_hist(array):
    cm = plt.cm.get_cmap('gist_rainbow')
    n, bins, patches = plt.hist(array, bins=64)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    plt.show()
plt.figure(figsize=(10,5))
plt.title("DIGIT: " + str(digit_a[-1]))
plt.xlabel("VOXEL")
plt.ylabel("POINTS INSIDE THE VOXEL")
plot_colorfull_hist(a_voxelgrid.structure[:,-1])

Train
with h5py.File("../input/train_small.h5", "r") as hf:
    size = len(hf.keys())
    # tuple to store the vectors and labels
    out = (np.zeros((size, 512), dtype="f"), np.zeros(size, dtype=np.int8))

    for i in range(size):
        if i % 200 == 0:
            print(i, "\t processed")
        voxelgrid = VoxelGrid(hf[str(i)]["points"][:], x_y_z=[8, 8, 8])
        # make the vector range 0-1
        out[0][i] = voxelgrid.vector.reshape(-1) / np.max(voxelgrid.vector)
        out[1][i] = hf[str(i)].attrs["label"]
    print("[DONE]")
    X_train, y_train = out

with h5py.File("../input/valid_small.h5", "r") as hf:
    size = len(hf.keys())
    # tuple to store the vectors and labels
    out = (np.zeros((size, 512), dtype="f"), np.zeros(size, dtype=np.int8))

    for i in range(size):
        if i % 200 == 0:
            print(i, "\t processed")
        voxelgrid = VoxelGrid(hf[str(i)]["points"][:], x_y_z=[8, 8, 8])
        # make the vector range 0-1
        out[0][i] = voxelgrid.vector.reshape(-1) / np.max(voxelgrid.vector)
        out[1][i] = hf[str(i)].attrs["label"]
    print("[DONE]")
    X_valid, y_valid = out

Train a simple linear model
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train)

Validation Score
print("Validation Score: ", clf.score(X_valid, y_valid))

# load images from HDF5 dataset
with h5py.File("../input/valid_small.h5", "r") as hf:
    wrong_pred = []
    for i in miss:
        wrong_pred.append((hf[str(i)]["img"][:], y_pred[i], y_valid[i]))
fig, axes = plt.subplots(2, 4, figsize=(10,10))

random_idx = np.random.randint(0, len(wrong_pred), 8)

for i, ax in enumerate(axes.flat):
    ax.imshow(wrong_pred[random_idx[i]][0])
    ax.set_title("PREDICTION: "+ str(wrong_pred[random_idx[i]][1]) + "\n" +
                "TRUE: " + str(wrong_pred[random_idx[i]][2]))
    ax.set_xticks([])
    ax.set_yticks([])

import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
array = confusion_matrix(y_valid, y_pred)
df_cm = pd.DataFrame(array, index = range(10),
                  columns = range(10))
plt.figure(figsize=(10,8))
sn.heatmap(df_cm, annot=True)
