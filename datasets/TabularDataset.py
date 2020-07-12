from __future__ import print_function, division
import torch
import pandas as pd
from torch.utils.data import Dataset


class TabularDataset(Dataset):

    def __init__(self, first_file_path, separator=None, preprocess=None):
        """
        Initialise the TabularDataset with different filepaths. These filepaths
        are all csv files constituting the datasets. Also initialise with the dataframe
        which is read in this case.
        Can have more file names -> second_file_path, third_file_path, fourth_file_path)
        :param first_file_path: Path of the first file in csv provided in string format
        :param separator: Separator for the file if any
        :param preprocess: This is the processor which is to be used for the data elements
        """
        self.first_file_path = first_file_path
        self.first_dataframe = pd.read_csv(first_file_path, sep=separator)
        self.preprocess = preprocess

    def __len__(self):
        """
        Returns the length of the data frame
        :return: Length of the dataframe
        """
        return len(self.first_dataframe)

    def __getitem__(self, idx):
        """
        This returns the data element given the index of the data set. It performs
        processing on the data element as per the preprocess passed in the dataloader
        :param idx: The index of the data element
        :return: Processed data element or sample in torch.tensor element
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.first_dataframe.iloc[idx, :]
        if self.preprocess:
            sample = self.preprocess(sample)
        return sample
