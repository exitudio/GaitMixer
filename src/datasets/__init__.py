from .gait import (
    CasiaBPose,
    OUMVLPDataset,
    CasiaQueryDataset
)


def dataset_factory(name):
    if name == "casia-b":
        return CasiaBPose
    elif name == "casia-b-query":
        return CasiaQueryDataset
    elif name == "OUMVLP":
        return OUMVLPDataset
    raise ValueError()
