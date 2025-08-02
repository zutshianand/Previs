import os
import sys
import types

# Ensure repository root is on the path so ``previs`` can be imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a lightweight stub for ``torch.utils.data.DataLoader`` so the test can
# run without the heavy ``torch`` dependency.
torch_mod = types.ModuleType("torch")
utils_mod = types.ModuleType("torch.utils")
data_mod = types.ModuleType("torch.utils.data")

class DataLoader:
    def __init__(self, dataset, num_workers=0, batch_size=None):
        self.dataset = dataset

    def __iter__(self):  # pragma: no cover - trivial
        for i in range(len(self.dataset)):
            yield self.dataset[i]


data_mod.DataLoader = DataLoader
utils_mod.data = data_mod
torch_mod.utils = utils_mod
sys.modules.setdefault("torch", torch_mod)
sys.modules.setdefault("torch.utils", utils_mod)
sys.modules.setdefault("torch.utils.data", data_mod)

from previs.dataloaders.MultiStreamDataLoader import MultiStreamDataLoader


class DatasetA:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [1]


class DatasetB:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [2]


def test_multistream_dataloader_combines_streams():
    loader = MultiStreamDataLoader([DatasetA(), DatasetB()])
    batch = next(iter(loader))
    assert batch == [1, 2]
