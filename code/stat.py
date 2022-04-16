#%%
import numpy as np
def calc_mean_and_std(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

train_data = np.load('data_cache/train_data_combined.npy')
print(calc_mean_and_std(train_data))
# %%
test_data = np.load('data_cache/test_data_combined.npy')
print(calc_mean_and_std(test_data))
# %%
