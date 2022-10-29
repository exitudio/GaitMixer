import numpy as np
from torch.utils.data import Dataset
import glob
from datasets.augmentation import PadSequence

class PoseDataset(Dataset):
    """
    Args:
     data_list_path (string):   Path to pose data.
     sequence_length:           Length of sequence for each data point. The number of frames of pose data returned.
     transform:                 Transformation on the dataset
    """

    def __init__(
        self,
        data_list_path,
        sequence_length=1,
        duplicate_bgcl=False,
        transform=None,
    ):
        super(PoseDataset, self).__init__()
        self.data_list = np.loadtxt(data_list_path, skiprows=1, dtype=str)
        self.sequence_length = sequence_length

        self.transform = transform

        self.data_dict = {}

        for idx, row in enumerate(self.data_list):
            row = row.split(",")

            # target = (subject_id, walking_status, sequence_num, view_angle)
            target, frame_num = self._filename_to_target(row[0])

            if target not in self.data_dict:
                self.data_dict[target] = {}

            if len(row[1:]) != 51:
                print("Invalid pose data for: ",
                      target, ", frame: ", frame_num)
                continue
            # Added try block to see if all the joint values are present. other wise skip that frame.
            try:
                self.data_dict[target][frame_num] = np.array(
                    row[1:], dtype=np.float32
                ).reshape((-1, 3))
            except ValueError:
                print("Invalid pose data for: ",
                      target, ", frame: ", frame_num)
                continue

            # if idx > 1000: break

        # Check for data samples that have less than sequence_length frames and remove them.
        for target, sequence in self.data_dict.copy().items():
            if len(sequence) < self.sequence_length + 1:
                del self.data_dict[target]
            else:
                if duplicate_bgcl and (target[1] == 1 or target[1] == 2):
                    for i in range(2):
                        new_target = [*target]
                        new_target[2] += (i+1) * 2
                        self.data_dict[tuple(new_target)] = self.data_dict[target]

        self.targets = list(self.data_dict.keys())
        self.data = list(self.data_dict.values())
        print('CasiaBPose:', len(self.targets))

    def _filename_to_target(self, filename):
        raise NotImplemented()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (pose, target) where target is index of the target class.
        """
        target = self.targets[index]
        data = np.stack(list(self.data[index].values()))
        if self.transform is not None:
            data = self.transform(data)

        return data, target

    def get_num_classes(self):
        """
        Returns number of unique ids present in the dataset. Useful for classification networks.

        """
        if type(self.targets[0]) == int:
            classes = set(self.targets)
        else:
            classes = set([target[0] for target in self.targets])
        num_classes = len(classes)
        return num_classes


class CasiaBPose(PoseDataset):
    """
    CASIA-B Dataset
    The format of the video filename in Dataset B is 'xxx-mm-nn-ttt.avi', where
      xxx: subject id, from 001 to 124.
      mm: walking status, can be 'nm' (normal), 'cl' (in a coat) or 'bg' (with a bag).
      nn: sequence number.
      ttt: view angle, can be '000', '018', ..., '180'.
     """

    mapping_walking_status = {
        'nm': 0,
        'bg': 1,
        'cl': 2,
    }

    def _filename_to_target(self, filename):
        _, sequence_id, frame = filename.split("/")
        subject_id, walking_status, sequence_num, view_angle = sequence_id.split(
            "-")
        walking_status = self.mapping_walking_status[walking_status]
        return (
            (int(subject_id), int(walking_status),
             int(sequence_num), int(view_angle)),
            int(frame[:-4]),
        )




class OUMVLPDataset(Dataset):
    def __init__(
        self,
        train_data_path,
        sequence_length=30,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.sequence_length = sequence_length
        self.paths = sorted(glob.glob(train_data_path+'/*/*.npy'))
        self.subjects = []
        for walk_paths in self.paths:
            subject = walk_paths.split('/')[-2]
            self.subjects.append(int(subject))
        self.pad_seq = PadSequence(sequence_length=sequence_length)

        # triplet sampler
        self.label_set = self.subjects
        # print(self.label_set)
        self.indices_dict = {}
        for i, label in enumerate(self.label_set):
            if not label in self.indices_dict.keys():
                self.indices_dict[label] = []
            else:
                self.indices_dict[label].append(i)
        # print(self.indices_dict)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = np.load(self.paths[index], allow_pickle=True)
        data = np.delete(data, 1, axis=1)
        data[:, :, :2] = data[:, :, :2]/3

        data = self.pad_seq(data)

        # transform
        if self.transform is not None:
            data = self.transform(data)

        subject = int(self.paths[index].split('/')[-2])
        view_seq = self.paths[index].split('/')[-1].split('.')[0].split('_')
        view_seq = [int(v) for v in view_seq]
        return data, [subject, *view_seq]



class CasiaQueryDataset(Dataset):
    def __init__(
        self,
        data_list_path,
        id_range,
        duplicate_bgcl=False,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        
        self.walk_paths = []
        _start, _end, *other = id_range.split('-')
        _start = int(_start)
        _end = int(_end)
        self.targets = []
        for walk_path in sorted(glob.glob(data_list_path+'/*')):
            subject_id = int(walk_path.split('/')[-1].split('_')[0])
            if subject_id >= _start and subject_id <= _end:
                self.walk_paths.append(walk_path)
                target = walk_path.split('/')[-1].split('.')[0].split('_')
                target = list(map(int, target))
                self.targets.append(target)

                if duplicate_bgcl and (target[1] == 1 or target[1] == 2):
                    for i in range(2):
                        self.walk_paths.append(walk_path)
                        self.targets.append(target)
        print('CasiaQueryDataset:', len(self.walk_paths))

    def __len__(self):
        return len(self.walk_paths)

    def __getitem__(self, index):
        data = np.load(self.walk_paths[index])
        if self.transform is not None:
            data = self.transform(data)
        target = self.walk_paths[index].split('/')[-1].split('.')[0].split('_')
        target = list(map(int, target))
        return data, target
