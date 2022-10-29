from .gait import (
    CasiaBPose,
    CasiaQueryDataset
)


def dataset_factory(name):
    if name == "casia-b":
        return CasiaBPose
    elif name == "casia-b-query":
        return CasiaQueryDataset
    raise ValueError()
