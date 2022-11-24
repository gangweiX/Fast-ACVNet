from .kitti_dataset_1215_augmentation import KITTIDataset
from .sceneflow_dataset_augmentation import SceneFlowDatset
from .middlebury_data_our import MiddleburyStereoDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "middlebury":MiddleburyStereoDataset
}
