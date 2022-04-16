import os
from typing import List, Tuple
import musdb
import librosa
import numpy as np
from enum import IntFlag
from torch.utils.data import Dataset
import logging

logger = logging.getLogger()

class TrackType(IntFlag):
    DRUMS = 1
    BASS = 2
    VOCALS = 4
    OTHER = 8

def get_audio(track:musdb.audio_classes.MultiTrack, chunk_start:float, chunk_duration:float, trackType:TrackType)->Tuple[np.ndarray, np.ndarray]:
    track.chunk_start = chunk_start
    track.chunk_duration = chunk_duration
    if trackType == trackType.DRUMS | trackType.BASS | trackType.VOCALS | trackType.OTHER:
        audio = track.audio.sum(axis=1)
    else:
        audio = np.zeros(track.audio.shape[0], dtype=np.float64)
        if trackType & TrackType.DRUMS:
            audio += track.targets['drums'].audio.sum(axis=1)
        if trackType & TrackType.BASS:
            audio += track.targets['bass'].audio.sum(axis=1)
        if trackType & TrackType.VOCALS:
            audio += track.targets['vocals'].audio.sum(axis=1)
        if trackType & TrackType.OTHER:
            audio += track.targets['other'].audio.sum(axis=1)
    audio = audio.astype(np.float32)
    label = np.array([trackType & trackType.DRUMS, trackType & trackType.BASS, trackType & trackType.VOCALS, trackType & trackType.OTHER], dtype=np.float32)
    label[label > 0] = 1
    return audio, label

def get_spec_img(track:musdb.audio_classes.MultiTrack, chunk_start:float, chunk_duration:float, trackType:TrackType)->Tuple[np.ndarray, np.ndarray]:
    audio, label = get_audio(track, chunk_start, chunk_duration, trackType)
    spec_img = librosa.feature.melspectrogram(y=audio, sr=track.rate, n_mels=64)
    return spec_img, label

def filter_no_sound_data(data, min_vol=10):
    return np.argwhere(data.max(axis=(1, 2)) <= min_vol).flatten()

def mark_data(marks, length, stride=15):
    mark = np.zeros(length, dtype=np.int)
    for i in marks:
        i -= i % stride
        for t in range(stride):
            mark[i + t] = 1
    return mark

class ClassificationDataset(Dataset):
    def __init__(self, root:str, subsets:List[str], chunk_duration=5.0, chunk_overlap=0.0, transform=None, target_transform=None, is_wav=False, use_cache=False, cache_dir = 'code/data_cache') -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.subsets = subsets
        self.cache_data_name = f'{self.cache_dir}/{self.subsets[0]}_data_combined.npy'
        self.cache_label_name = f'{self.cache_dir}/{self.subsets[0]}_label_combined.npy'
        if self.use_cache:
            logger.info(f'Loading cached data from {self.cache_data_name}')
            self.cache_data = np.load(self.cache_data_name)
            self.cache_label = np.load(self.cache_label_name)
            mask = mark_data(filter_no_sound_data(self.cache_data), self.cache_data.shape[0])
            self.cache_data = self.cache_data[mask == 0]
            self.cache_label = self.cache_label[mask == 0]
            self.cache_label = self.cache_label[:, :3]
            self.len = len(self.cache_data)
            print(f'dataset len: {self.len}')

        else:
            self.mus= musdb.DB(root=root, subsets=subsets, is_wav=is_wav)
            self.chunk_duration = chunk_duration
            self.chunk_overlap = chunk_overlap
            self.start_pos = [[i for i in np.arange(0.0, track.duration - self.chunk_duration, self.chunk_duration - self.chunk_overlap)] for track in self.mus.tracks]
            self.len = sum(len(i) for i in self.start_pos) * 15

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx:int) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_cache:
            spec = self.cache_data[idx]
            label = self.cache_label[idx]
        else:
            trackType = TrackType(idx % 15 + 1)
            pos = idx // 15
            track = 0
            while pos >= len(self.start_pos[track]):
                pos -= len(self.start_pos[track])
                track += 1
            spec, label = get_spec_img(self.mus.tracks[track], pos, self.chunk_duration, trackType)
        if self.transform is not None:
            spec = self.transform(spec)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return spec, label

class GenerationDataset(Dataset):
    def __init__(self, root, subsets, chunk_duration=5.0, chunk_overlap=0.0, transform=None, target_transform=None, is_wav=False, use_cache=False) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.cache_dir = 'code/data_cache'
        self.use_cache = use_cache
        self.subsets = subsets
        self.cache_data_name = f'{self.cache_dir}/{self.subsets[0]}_data_combined.npy'
        self.cache_label_name = f'{self.cache_dir}/{self.subsets[0]}_label_combined.npy'
        if self.use_cache:
            logger.info(f'Loading cached data from {self.cache_data_name}')
            self.cache_data = np.load(self.cache_data_name)
            self.cache_label = np.load(self.cache_label_name)
            self.len = len(self.cache_data)
        else:
            self.mus= musdb.DB(root=root, subsets=subsets, is_wav=is_wav)
            self.chunk_duration = chunk_duration
            self.chunk_overlap = chunk_overlap
            self.start_pos = [[i for i in np.arange(0.0, track.duration - self.chunk_duration, self.chunk_duration - self.chunk_overlap)] for track in self.mus.tracks]
            self.len = sum(len(i) for i in self.start_pos) * 15
        

class PartialDataset(ClassificationDataset):
    def __init__(self, root:str, subsets:List[str], chunk_duration=5.0, chunk_overlap=0.0, transform=None, target_transform=None, is_wav=False, use_cache=False, start_at=None, end_at=None) -> None:
        super().__init__(root, subsets, chunk_duration, chunk_overlap, transform, target_transform, is_wav, use_cache)
        self.start = start_at if start_at is not None else 0
        self.end = end_at if end_at is not None else self.len
        self.cache_data = self.cache_data[self.start:self.end]
        self.cache_label = self.cache_label[self.start:self.end]
        self.len = self.end - self.start

if __name__ == '__main__':
    import tqdm
    from torch.utils.data import DataLoader
    train_dataset = ClassificationDataset(root='./datasets/musdb18', subsets=['train'])
    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    test_dataset = ClassificationDataset(root='./datasets/musdb18', subsets=['test'])
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8)
    print(len(train_dataset), len(test_dataset))
    for i in tqdm.tqdm(testloader):
        pass
