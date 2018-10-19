import torch.utils.data as data
from PIL import Image


class CustomDataset(data.Dataset):
    """
    This gets elements from a numpy array and wraps it up into a Dataset object.
    It aids faster loading while sampling datapoints in lib.samplers.sampler
    """
    def __init__(self, data, transforms):
        self.data = data
        self.trasforms = transforms

    def __getitem__(self, index):
        img = self.data[index]

        img = Image.fromarray(img)
        if self.trasforms is not None:
            img = self.trasforms(img)

        return img

    def __len__(self):
        return len(self.data)
