import torch
import labels
import numpy as np

class Dataset:
    def __init__(self, images, image_labels):
        self.images = torch.from_numpy(images).float() if isinstance(images, np.ndarray) else images
        self.image_labels = image_labels

        self.num_labels = labels.n_base

        self.labels_onehot = torch.eye(self.num_labels)
        self.diacritic_onehot = torch.eye(3) # none, dakuten, handakuten
        self.script_onehot = torch.eye(2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.image_labels[idx].item() # 0 - 144
        script = self.script_onehot[0 if label <= 70 else 1]


        if label in labels.dakuten_to_base:
            base_equivalent = labels.dakuten_to_base[label]

            return image, (script, self.labels_onehot[base_equivalent], self.diacritic_onehot[1])

        if label in labels.handakuten_to_base:
            base_equivalent = labels.handakuten_to_base[label]

            return image, (script, self.labels_onehot[base_equivalent], self.diacritic_onehot[2])

        return image, (script, self.labels_onehot[label], self.diacritic_onehot[0])