import numpy as np
import torch
import random
import math



class ToTensor(object):
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float)




class FlipSequence(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            return np.flip(data, axis=0).copy()
        return data


class MirrorPoses(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            center = np.mean(data[:, :, 0], axis=1, keepdims=True)
            data[:, :, 0] = center - data[:, :, 0] + center
            # data[:, :, 0] = 320 - data[:, :, 0]

        return data


class RandomSelectSequence(object):
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            if data.shape[0] == self.sequence_length:
                return data
            start = np.random.randint(0, data.shape[0] - self.sequence_length)
        except ValueError:
            print('RandomSelectSequence error length', data.shape[0])
            raise ValueError
        end = start + self.sequence_length
        return data[start:end]


class SelectSequenceCenter(object):
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = int((data.shape[0]/2) - (self.sequence_length / 2))
        except ValueError:
            print('SelectSequenceCenter error lengtth:', data.shape[0])
            raise ValueError
        end = start + self.sequence_length
        return data[start:end]


class TwoNoiseTransform(object):
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x1, x2=None):
        if x2 is not None:
            # for multi camera view from the same subject id (using multiview as augmentation)
            return [self.transform(x1), self.transform(x2)]
        return [self.transform(x1), self.transform(x1)]


class ThreeCenterSequenceTransform(object):
    """Create two crops of the same image"""

    def __init__(self, transform, sequence_length=60):
        self.transform = transform
        self.sequence_length = sequence_length

    def __call__(self, data):
        gap = int(self.sequence_length/3)
        start = int((data.shape[0]/2) - (self.sequence_length / 2))
        end = start + self.sequence_length
        left_start = max(start-gap, 0)
        right_end = min(end+gap, data.shape[0])

        data1 = data[left_start:left_start+self.sequence_length]
        data2 = data[start:end]
        data3 = data[right_end-self.sequence_length:right_end]
        return [self.transform(data1), self.transform(data2), self.transform(data3)]


class PointNoise(object):
    """
    Add Gaussian noise to pose points
    std: standard deviation
    """

    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
        return data + noise


class JointNoise(object):
    """
    Add Gaussian noise to joint
    std: standard deviation
    """

    def __init__(self, std=0.5):
        self.std = std

    def __call__(self, data):
        # T, V, C
        noise = np.hstack((
            np.random.normal(0, 0.25, (data.shape[1], 2)),
            np.zeros((data.shape[1], 1))
        )).astype(np.float32)

        return data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)


def normalize_width(data):
    data = data.copy()
    data[:, :, :2] = data[:, :, :2] / 320
    return data

def remove_conf(enable):
    def _rm_conf(data):
        if enable:
            return data[:, :, :2]
        return data
    return _rm_conf

def joint_drop(data):
    j = 17
    if random.random() < .9:
        f=data.shape[0]
        num_random = int(f*j*.2)
        num_total = f*j
        addition_noise = 50+np.random.rand(num_random*2)*100 # rand [10,50] or [-10, 50] frame height is 980 but /3 when preprocess
        addition_noise *= np.random.randint(0,2, size=(num_random*2))*2-1
        addition_noise = addition_noise.reshape((num_random,2))
        addition_noise = np.concatenate((addition_noise, np.random.rand(num_random,1)), axis=1)

        index_drop = np.random.default_rng().choice(num_total, size=num_random, replace=False)
        data = data.reshape((-1, 3))
        data[index_drop, :2] += addition_noise[:, :2]
        data[index_drop, 2] = addition_noise[:, 2]
        data = data.reshape((-1, j, 3))
    return data
    

class PadSequence:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length

    def __call__(self, data):
        input_length = data.shape[0]
        if input_length > self.sequence_length:
            return data
        
        diff = self.sequence_length + 1 - input_length
        len_pre = int(math.ceil(diff / 2))
        len_pos = int(diff / 2) + 1

        while len_pre > data.shape[0] or len_pos > data.shape[0]:
            data = np.concatenate([np.flip(data, axis=0), 
                                   data, 
                                   np.flip(data, axis=0)], 
                                  axis=0)
        pre = np.flip(data[1:len_pre], axis=0)
        pos = np.flip(data[-1 - len_pos:-1], axis=0)
        data = np.concatenate([pre, data, pos], axis=0)[:self.sequence_length]
        return data

def drop_arm(data):
    if np.random.random() <= .1:
        # 7,9  8,10
        if random.randint(0, 1) == 0:
            elbow = 7
            hand = 9
        else:
            elbow = 8
            hand = 10
        data[:, elbow] = [0, 0, 0]
        data[:, hand] = [0, 0, 0]
    return data