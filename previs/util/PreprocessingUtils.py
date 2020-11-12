from os import listdir
from os.path import isfile, join

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


def merge_lists(first_list, second_list):
    """This method merges the two lists input here
    in one lists such that the resulting list contains tuples containing the elements
    of the two lists.
    @param first_list: The first list
    @param second_list: The second list
    @return: [(el11, el21), (el12, el22)] a list of tuples
    """
    final_list = []

    for i in range(len(first_list)):
        final_list.append((first_list[i], second_list[i]))

    return final_list


def build_text_file_from_cls(dir_path, cls):
    """This method does the following.
    Given a directory containing only text files, we consider that the directory
    containts sub directories having names of a class cls. So the subdirectory's name is
    <dir_path/cls>
    This directory contains different .txt files. We read all these files into different
    lists of strings and output the resultant list of lists:
    [
    ["fgbfggbf fhgfdgf fbfg"],
    ["dfbdf fgnhgfh gfdg"]
    ]
    @param dir_path: The root directory path as explained above
    @param cls: The name of the class in int or string. It will also be the name of the subdirectory
    as explained above
    @return: List of list of strings containing the input of the text file in one list and others in
    the next one
    """
    text_file_list = []
    sub_dir_path = dir_path + "/" + str(cls)
    for f in listdir(sub_dir_path):
        if isfile(join(sub_dir_path, f)):
            file_path = join(sub_dir_path, f)
            file = open(file_path, 'r')
            res_line = ""
            for line in file.readlines():
                res_line += line.strip() + " "
            text_file_list.append([res_line.strip()])

    return text_file_list
