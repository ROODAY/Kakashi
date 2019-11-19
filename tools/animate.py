# cannibalize the functions that are needed from videopose. This file should be a standalone file that can be dropped into videpose root

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

print('truth')
data = np.load('data/test/00001/00001.keypoints.npy')
frame = data[1000]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [dot[0] for dot in frame]
y = [dot[1] for dot in frame]
z = [dot[2] for dot in frame]

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

print('estimated')
data = np.load('out/00001.keypoints.npy')
frame = data[1000]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [dot[0] for dot in frame]
y = [dot[1] for dot in frame]
z = [dot[2] for dot in frame]

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()