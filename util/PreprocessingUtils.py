import numpy as np
import torch
import torch.nn.functional as F


def one_hot_encode_column(max_class, resultant_sample, sample, column_name):
    """
    :param max_class: The max value of the categorical data which is now ordinal in nature
    The values are staring from 0 and ending to max_class
    :param resultant_sample: Numpy array in which you want to append the converted sample
    :param sample: The data Series which needs to be converted using one-hot-encoding
    :param column_name:
    :return: The one-hot-encoded value appended to the resultant_sample
    """
    column_values = F.one_hot(torch.from_numpy(np.array(sample[column_name])),
                              max_class).numpy()[np.newaxis]
    return np.append(resultant_sample, column_values)
