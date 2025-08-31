import time

import joblib
import numpy as np
from PIL import ImageFilter
from sklearn.tree._criterion import MSE
from torch.nn.functional import one_hot
import torch
import labels
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

from model import KanaModel

import matplotlib.pyplot as plt

def view_single_image(image_array, char, index=0):
    """
    View a single image from your dataset
    image_array: numpy array of shape (num_images, 48, 48) with values 0-1
    index: which image to display
    """
    # Get the specific image
    img = image_array

    # Create plot
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Image {index} - {char}")
    plt.axis('off')  # Hide axes
    plt.show()

    return img

train_images_hira = np.load("hiragana_test_images.npz")['arr_0']
train_labels_hira = np.load("hiragana_test_labels.npz")['arr_0']
train_images_kata = np.load("katakana_test_images.npz")['arr_0']
train_labels_kata = np.load("katakana_test_labels.npz")['arr_0']

# label each label in the group with 0 or 1 (hiragana/katakana)
# [0, 1, 2, 3] -> [(0, 0), (0, 1), ...]
train_labels_hira = np.column_stack((np.zeros_like(train_labels_hira), train_labels_hira))
train_labels_kata = np.column_stack((np.ones_like(train_labels_kata), train_labels_kata))

test_images = np.concatenate([train_images_hira, train_images_kata])
test_labels = np.concatenate([train_labels_hira, train_labels_kata])

dataset = Dataset(test_images, test_labels)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



total = len(dataset)

model1 = KanaModel(in_channels=1, d_model=64, out_base=labels.n_base, out_dakuten=3)
model2 = KanaModel(in_channels=1, d_model=64, out_base=labels.n_base, out_dakuten=3)

model1.load_state_dict(torch.load("models/KanaNet_e20.pth"))
model2.load_state_dict(torch.load("models/KanaNet_e25.pth"))


correct = [0, 0]
script_onehot = torch.eye(2)
for i, model in enumerate([model1, model2]):
    for idx, (batch_x, (y_base, y_diacritic)) in enumerate(dataloader):
        (x_script, x_image) = batch_x
        with torch.no_grad():
            model.eval()
            pred_base, pred_diacritic = model(x_script, x_image.unsqueeze(0))

        if y_base.argmax() == pred_base.argmax() and y_diacritic.argmax() == pred_diacritic.argmax():
            correct[i] += 1

        if idx % 100 == 0:
            print(f"{idx} / {total} done")

    print("model done")


print(f"Accuracy 1: {100 * correct[0] / total}")
print(f"Accuracy 2: {100 * correct[1] / total}")