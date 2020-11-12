from os import listdir
from os.path import isfile, join

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, image_dir_path, classification_format=False, transform=None):
        """
        We initialise the ImageDataset with the following values
        :param image_dir_path: This is the folder path where all the images are stored.
        The folder will only have image files ending with suitable extension
        :param classification_format: This is true when the directories are in the format
        of a classification problem. See README.md for more details on this
        :param transform: Composition of transformation to be applied sequentially are passed here.
        User has to determine the transformations and pass it here. Refer to README.md for details.
        """
        self.image_dir_path = image_dir_path
        self.classification_format = classification_format
        self.transform = transform

        self.image_file_paths = []
        [self.image_file_paths.append(join(self.image_dir_path, f))
         for f in listdir(self.image_dir_path)
         if isfile(join(self.image_dir_path, f))]

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
        image = Image.open(open(image_path, 'rb'))
        if self.transform:
            image = self.transform(image)
        return image
