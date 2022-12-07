# %%
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from a5_utils import *

# %% [markdown]
# # Exercise 1
# 
# ### 1a
# 
# disparity: d = x1-x2
# 
# x1 / f = px / pz,
# 
# -x2 / f = (T - px) / pz
# 
# => x1 = px / pz * f
# 
# => -x2 = f * T / pz - f * px / pz
# 
# d = x1-x2 = f * px / pz - f * px / pz + f * T / px = f * T / pz
# 
# T and f constant => bigger pz smaller disparity smaller pz bigger disparity

# %% [markdown]
# ### 1b

# %%
def disparity(pz, T, f):
    return T * f / pz

T = 0.0025
f = 0.12
pzArr = np.arange(0.1, 10, 0.1) # start, stop, step
disparitiesArr = disparity(pzArr, T, f)

fig, a = plt.subplots(1, 1)
fig.set_figwidth(5)
fig.set_figheight(5)
a.set_title("Disparity - distance")
a.set_xlabel("distance[m]")
a.set_ylabel("disparity[m]")
a.set(xlim=(0, 10), ylim=(0, 0.003))
a.plot(pzArr, disparitiesArr)
plt.show()



