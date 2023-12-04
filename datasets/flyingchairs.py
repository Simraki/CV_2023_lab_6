from .customdataset import CustomDataset
from .utils import make_dataset_split, get_flying_chairs_data_paths


def flying_chairs(
        dataset_dir: str,
        transform=None,
        target_transform=None,
        split: float = 0.8,
        seed: int = 42
):
    data_paths = get_flying_chairs_data_paths(dataset_dir)
    train_list, test_list = make_dataset_split(data_paths, split, seed)
    train_dataset = CustomDataset(file_names=train_list, transform=transform, target_transform=target_transform)
    test_dataset = CustomDataset(file_names=test_list, transform=transform, target_transform=target_transform)

    return train_dataset, test_dataset
