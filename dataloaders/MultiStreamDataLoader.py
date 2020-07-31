from itertools import chain

from torch.utils.data import DataLoader


class MultiStreamDataLoader:

    def __init__(self, datasets):
        """
        @param datasets: These are the different types of datasets which
        are used to make the multi stream data loader. Refer to Readme.md for more details
        on this.
        """
        self.datasets = datasets

    def get_stream_loaders(self):
        return zip(*[DataLoader(dataset, num_workers=1, batch_size=None)
                     for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield list(chain(*batch_parts))
