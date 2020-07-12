import numpy as np

from util.PreprocessingUtils import one_hot_encode_column


class TabularProcessor(object):
    """
    Applies processing on a data row
    The data row can constitute of the following:
    1. Categorical data
    2. Numerical data
    """

    def __init__(self, columns_to_encode=None, max_encoded_values=None):
        """
        Applies any form of processing to the sample and returns it. As of now,
        now pre-processing is being done here because we are assuming that the
        data here is numerical in nature and does not require any pre-processing
        :param columns_to_encode: List of column names which need to be one-hot-encoded.
        The columns should have ordinal data beforehand.
        :param max_encoded_values: List of the maximum ordinal value in each of these column names
        """
        self.columns_to_encode = columns_to_encode
        self.max_encoded_values = max_encoded_values

    def __call__(self, sample):
        """
        This method returns the sample in the processed format. The processing can be modified
        as per the requirement. For this example and processor, we are considering the sample
        to have only ordinal and numerical values.
        :param sample: A pandas Series having the values
        :return: A numpy array which contains the processed data
        """
        resultant_sample = np.array([])
        idx = 0
        if self.columns_to_encode is not None:
            for column_name in sample.index:
                if column_name not in self.columns_to_encode:
                    column_values = list(np.array(sample[column_name])[np.newaxis])
                    resultant_sample = np.append(resultant_sample, column_values)
                else:
                    num_class = self.max_encoded_values[idx]
                    idx += 1
                    resultant_sample = one_hot_encode_column(num_class, resultant_sample, sample, column_name)
        resultant_sample = np.array(resultant_sample)
        return resultant_sample
