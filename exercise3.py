# %%
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from a5_utils import *

# %% [markdown]
# # Exercise 3

# %%
temp = cv2.imread('data/epipolar/house1.jpg') # 0-255
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
temp = temp.astype(np.float64) / 255
temp2 = cv2.imread('data/epipolar/house2.jpg') # 0-255
temp2 = cv2.cvtColor(temp2, cv2.COLOR_BGR2GRAY)
temp2 = temp2.astype(np.float64) / 255
data = np.loadtxt("data/epipolar/house_points.txt")
P1 = np.loadtxt("data/epipolar/house1_camera.txt")
P2 = np.loadtxt("data/epipolar/house2_camera.txt")
print(data)
print(P1)
print(P2)

# %%
points = np.array([])
A = np.array([])
for d in data:
    x1 = d[0:2]
    x2 = d[2:4]
    x1x = np.array([[0, -1, x1[1]], [1, 0, -x1[0]], [-x1[1], x1[0], 0]])
    x2x = np.array([[0, -1, x2[1]], [1, 0, -x2[0]], [-x2[1], x2[0], 0]])
    a1 = np.matmul(x1x, P1)[0:2, :]
    a2 = np.matmul(x2x, P2)[0:2, :]
    A = a1
    A = np.vstack((A, a2))
    U, D, VT = np.linalg.svd(A)
    x = VT.T[:,-1] / VT[-1,-1]
    if points.size == 0:
        points = x
    else:
        points = np.vstack((points, x))
        
print(points)



# %%
f = plt.figure(figsize=(15, 10))
f.add_subplot(1, 3, 1)
plt.imshow(temp, cmap='gray')

for i, d in enumerate(data):
    x1 = d[0]
    y1 = d[1]
    plt.plot(x1, y1, 'bo', markersize=3)
    plt.text(x1, y1, i)

f.add_subplot(1, 3, 2)
plt.imshow(temp2, cmap="gray")

for i, d in enumerate(data):
    x1 = d[2]
    y1 = d[3]
    plt.plot(x1, y1, 'bo', markersize=3)
    plt.text(x1, y1, i)

ax = f.add_subplot(1, 3, 3, projection = '3d')

T = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

for i, d in enumerate(points):
    d = T.dot(d[0:3])
    print(d)
    x1 = d[0]
    y1 = -d[1]
    z1 = -d[2]
    ax.plot(x1, y1, z1, 'bo', markersize=3)
    ax.text(x1, y1, z1, s = i)

plt.show()


