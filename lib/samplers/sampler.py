import torch
from datasets.custom_dataset import CustomDataset

import numpy as np

class Sampler(object):
    def __init__(self, model, transforms, set, subset_size):
        model.eval()
        self.set = set
        self.subset_size = subset_size
        self.final_activations = []
        self.penultimate_activations = []
        self._get_activations(model,transforms)

    def _get_activations(self, model, transforms):
        dataset = CustomDataset(self.set, transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, sampler=None,
                                             batch_sampler=None, num_workers=10)
        for img in loader:
            final_acts, penultimate_acts = model(img)
            self.final_activations.extend(final_acts.detach().cpu().numpy())
            self.penultimate_activations.extend(penultimate_acts.detach().cpu().numpy())

    def get_subset(self):
        raise NotImplementedError
