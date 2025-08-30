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

def view_single_image(image_array, index=0):
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
    plt.title(f"Image {index}")
    plt.axis('off')  # Hide axes
    plt.show()

    return img

test_images_hira = np.load("hiragana_test_images.npz")['arr_0']
test_images_kata = np.load("katakana_test_images.npz")['arr_0']
test_labels_hira = np.load("hiragana_test_labels.npz")['arr_0']
test_labels_kata = np.load("katakana_test_labels.npz")['arr_0']

test_labels_kata = np.array([label + len(labels.hiragana) for label in test_labels_kata])
test_images = np.concatenate([test_images_hira, test_images_kata])
test_labels = np.concatenate([test_labels_hira, test_labels_kata])

# for i in test_images_hira:
#     print(i)
#     view_single_image(i)


dataset = Dataset(test_images, test_labels)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



total = len(dataset)

model1 = KanaModel(in_channels=1, d_model=64, out_base=labels.n_base, out_dakuten=3)
model2 = KanaModel(in_channels=1, d_model=64, out_base=labels.n_base, out_dakuten=3)

model1.load_state_dict(torch.load("models/KanaNet_e5.pth"))
model2.load_state_dict(torch.load("models/KanaNet_e20.pth"))


incorrect_script = [0, 0]
correct = [0, 0]

for i, model in enumerate([model2]):
    for idx, (batch_x, (y_script, y_base, y_diacritic)) in enumerate(dataloader):
        batch_x = batch_x.unsqueeze(1)

        with torch.no_grad():
            model.eval()
            pred_script, pred_base, pred_diacritic = model(batch_x)

        if pred_script.argmax() != y_script.argmax():
            incorrect_script[i] += 1
            print("incorrect script guess for idx", idx)

        if pred_script.argmax() == y_script.argmax() and pred_diacritic.argmax() == y_diacritic.argmax() and pred_base.argmax() == y_base.argmax():
            correct[i] += 1

        if idx % 100 == 0:
            print(f"{idx} / {total} done")

    print("model done")


print(f"Accuracy 1: {100 * correct[0] / total}")
print(f"Accuracy 2: {100 * correct[1] / total}")
print("incorrectly guessed script 1:", incorrect_script[0])
print("incorrectly guessed script 2:", incorrect_script[1])
