from typing import Callable, Optional

from torch.utils.data import Dataset

from .utils import flying_chairs_loader


class CustomDataset(Dataset):
    def __init__(
            self, file_names: [str],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader=flying_chairs_loader
    ):
        """
        Initialized the custom datasets
        :param file_names: list of absolute paths to samples along with path to their targets
        :param transform: transformation for inputs
        :param target_transform: transformation for target
        :param loader: a helper function to read the inputs and targets into numpy array
        """
        self.file_names = file_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, idx):
        """
        will return the sample with index idx from the dataset
        :param idx: the index of the sample which will be returned
        :return: input, target pair
        """
        inputs, target = self.loader(self.file_names[idx])
        # apply the transforms and target transform in case they exist
        if self.transform:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        """
        will return the total number of samples in the dataset.
        :return: number of samples in the dataset
        """
        return len(self.file_names)
