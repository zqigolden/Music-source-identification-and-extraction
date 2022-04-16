from pathlib import Path
import numpy as np
batch_size = 100

path = Path('code/data_cache')
train_data = sorted([i for i in path.glob('train_data_*')])
train_labels = sorted([i for i in path.glob('train_label_*')])
test_data = sorted([i for i in path.glob('test_data_*')])
test_labels = sorted([i for i in path.glob('test_label_*')])

def combine(data):
    data = np.concatenate([np.load(i)[None, ...] for i in data], axis=0)
    return data

np.save(str(path / 'train_data_combined.npy'), combine(train_data))
np.save(str(path / 'train_label_combined.npy'), combine(train_labels))
np.save(str(path / 'test_data_combined.npy'), combine(test_data))
np.save(str(path / 'test_label_combined.npy'), combine(test_labels))
