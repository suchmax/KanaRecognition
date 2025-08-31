import torch
import labels
import numpy as np

class Dataset:
    def __init__(self, images, image_labels):
        self.images = torch.from_numpy(images).float() if isinstance(images, np.ndarray) else images
        self.image_labels = image_labels

        self.base_onehot = torch.eye(labels.n_base)
        self.diacritic_onehot = torch.eye(3) # none, dakuten, handakuten
        self.script_onehot = torch.eye(2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        script, label = self.image_labels[idx] # (0|1, 0-45)
        script, label = script.item(), label.item()

        if label in labels.dakuten_to_base:
            base_equivalent = labels.dakuten_to_base[label]

            return (self.script_onehot[script], image), (self.base_onehot[base_equivalent], self.diacritic_onehot[1])

        if label in labels.handakuten_to_base:
            base_equivalent = labels.handakuten_to_base[label]

            return (self.script_onehot[script], image), (self.base_onehot[base_equivalent], self.diacritic_onehot[2])

        return (self.script_onehot[script], image), (self.base_onehot[label], self.diacritic_onehot[0])