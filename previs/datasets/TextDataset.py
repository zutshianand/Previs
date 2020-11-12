import random
from itertools import chain

from torch.utils.data.dataset import IterableDataset

from previs.util.PreprocessingUtils import merge_lists


class TextDataset(IterableDataset):

    def __init__(self, data_list, batch_size, transform=None,
                 maintain_order=False, predict_list=None):
        """
        @param data_list: List of list of strings. Refer to README.md for more details
        @param batch_size: Batch size
        @param transform: The transformation class which is used here. We use the TextProcessor
        @param maintain_order: This is used to determine whether it is a many-to-one or many-to-many problem
        we are trying to solve
        @param predict_list: In case of many-to-many problem, this is the list of list of tags
        """
        self.data_list = data_list
        self.batch_size = batch_size
        self.transform = transform
        self.maintain_order = maintain_order
        self.predict_list = predict_list

        if self.maintain_order:
            self.data_list = merge_lists(self.data_list, self.predict_list)

    @property
    def shuffled_data_list(self):
        """
        @return: Shuffles the data list randomly
        """
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        """This method is used to preprocess and apply cleaning to the textual data.
        The user can also apply different pretrained models as well here to generate embeddings.
        However, for simplicity and flexibility, we have skipped that as of now.
        @param data: The preprocessed data is generated in this case along with the respective tag
        if it is many-to-many problem
        """
        global processed_data
        data_to_be_processed = data[0]
        processed_data = data_to_be_processed

        if self.transform and self.maintain_order is not None:
            processed_data = self.transform(data_to_be_processed)

        processed_data_list = processed_data[0].split()
        for i in range(len(processed_data_list)):
            if self.predict_list:
                yield processed_data_list[i], data[1][i]
            else:
                yield processed_data_list[i]

    def get_stream(self, data_list):
        """
        @param data_list: This is the list of lists of strings
        @return: Return a chain object whose .__next__() method returns elements after
        getting processed
        """
        return chain.from_iterable(map(self.process_data, data_list))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list)
                     for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

    @classmethod
    def split_datasets(cls, data_list, batch_size,
                       max_workers, transform=None,
                       maintain_order=False, predict_list=None):
        """
        @param data_list: The list of list of strings of text
        @param batch_size: The batch size
        @param max_workers: The maximum number of cpu workers
        @param transform: The transformation class which is used here. We use the TextProcessor
        @param maintain_order: This is used to determine whether it is a many-to-one or many-to-many problem
        we are trying to solve
        @param predict_list: In case of many-to-many problem, this is the list of list of tags
        @return: It returns the TextDataset for the different batches after splitting them accordingly
        """
        global num_workers
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break

        split_size = batch_size // num_workers

        return [cls(data_list, split_size,
                    transform, maintain_order, predict_list)
                for _ in range(num_workers)]
