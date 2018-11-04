import torch
from datasets.custom_dataset import CustomDataset
import numpy as np

class Sampler(object):
    def __init__(self, model, dataset):
        model.eval()
        self.set = set
        self.dataset = dataset
        self.final_activations = []
        self.penultimate_activations = []
        self._get_activations(model)

    def _get_activations(self, model):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=500, shuffle=False, sampler=None,
                                             batch_sampler=None, num_workers=10)
        for img in loader:
            #print type(img)
            final_acts, penultimate_acts = model(img[0])
            #final_acts = model(img[0])
            self.final_activations.extend(final_acts.detach().cpu().numpy())
            self.penultimate_activations.extend(penultimate_acts.detach().cpu().numpy())

    def get_subset(self):
        raise NotImplementedError
