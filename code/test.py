#%%
import numpy as np

data:np.ndarray = np.load('data_cache/train_data_combined.npy')
print(data[0].shape)
#%%
def filter_no_sound_data(data, min_vol=0.1):
    return np.argwhere(data.max(axis=(1, 2)) <= min_vol).flatten()

def mark_data(marks, length, stride=15):
    mark = np.zeros(length, dtype=np.int)
    for i in marks:
        i -= i % stride
        for t in range(stride):
            mark[i + t] = 1
    return mark

data = data[:1000, ...]
mask = mark_data(filter_no_sound_data(data), data.shape[0])
data[mask == 0].shape
# %%
from IPython.display import display
import matplotlib.pyplot as plt
img = data[6002, ...].copy()
#img[img > 0.1] = 0.1
print(img.min())
plt.imshow(img)
# %%
quite_data = []
for i, v in enumerate(data):
    if v.max() < 10.0:
        quite_data.append(i)
len(quite_data)
# %%


data:np.ndarray = np.load('data_cache/train_data_combined.npy')

def filter_no_sound_data(data, min_vol=0.1):
    return np.argwhere(data.max(axis=(1, 2)) <= min_vol)

def mark_data(marks, length, stride=15):
    mark = [0] * length
    for i in marks:
        i -= i % stride
        for t in range(stride):
            mark[i + t] = 1
    return mark

mark_data([1, 5, 7, 8], 9, 3)

filter_no_sound_data(data)
# %%
# %%
