import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler
from imblearn.over_sampling import RandomOverSampler

def get_weighted_random_sampler(dataset):
    sampler_weights = torch.zeros(len(dataset))
    class_sampler = dataset.label_counts()
    for idx, label in enumerate(dataset.y):
        sampler_weights[idx] = class_sampler[label]

    sampler_weights = 1000. / sampler_weights
    return WeightedRandomSampler(sampler_weights.type('torch.DoubleTensor'), len(sampler_weights))

def get_oversampler(dataset):
    ros = RandomOverSampler(random_state=0)
    indices = np.arange(len(dataset))
    indices, _ = ros.fit_resample(indices.reshape(-1, 1), dataset.y)
    indices = indices.reshape(-1, )
    np.random.shuffle(indices)
    return SubsetRandomSampler(indices)
