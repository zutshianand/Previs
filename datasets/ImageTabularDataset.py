from os import listdir
from os.path import join, isfile

import pandas as pd

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageTabularDataset(Dataset):

    def __init__(self, image_dir_path, tabular_dir_path,
                 classification_format=False, image_transform=None, tabular_transform=None, separator=None):
        """
        This initialises the respective variables for both image directory and the
        tabular data csv file. For every row of the csv file, we have an image in the
        image directory.
        :param image_dir_path: This contains all the images in suitable image format
        :param tabular_dir_path: This is the path to the csv file
        :param classification_format: This is true when the directories are in the format
        of a classification problem. See Readme.md for more details on this
        :param image_transform: This is the composite transform for the images in the dataset
        :param tabular_transform: This is the composite transform for the csv rows in the dataset
        :param separator: This is the separator in the csv file used to read the dataset
        """
        self.image_dir_path = image_dir_path
        self.tabular_dir_path = tabular_dir_path
        self.classification_format = classification_format
        self.image_transform = image_transform
        self.tabular_transform = tabular_transform

        self.image_file_paths = []
        [self.image_file_paths.append(join(self.image_dir_path, f))
         for f in listdir(self.image_dir_path)
         if isfile(join(self.image_dir_path, f))]

        self.tabular_data = pd.read_csv(tabular_dir_path, sep=separator)

    def __len__(self):
        """
        This simple returns the length of the dataset. Essential for iterating the dataset.
        :return: Length of the dataset.
        """
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        """
        This returns the processed element with respect to the idx passed.
        :param idx: The position of the data element which needs processing
        :return: Processed data element as per the transforms set
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_file_paths[idx]
        sample = self.tabular_data.iloc[idx, :]
        image = Image.open(open(image_path, 'rb'))
        if self.image_transform:
            image = self.image_transform(image)
        if self.tabular_transform:
            sample = self.tabular_transform(sample)
        return image, sample
