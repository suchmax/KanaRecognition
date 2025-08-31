import time

from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import labels
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

from model import KanaModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = "Hiragino Sans"

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

# 48 x 48 (2304)
train_images_hira = np.load("hiragana_train_images.npz")['arr_0']
train_labels_hira = np.load("hiragana_train_labels.npz")['arr_0']
train_images_kata = np.load("katakana_train_images.npz")['arr_0']
train_labels_kata = np.load("katakana_train_labels.npz")['arr_0']

# label each label in the group with 0 or 1 (hiragana/katakana)
# [0, 1, 2, 3] -> [(0, 0), (0, 1), ...]
train_labels_hira = np.column_stack((np.zeros_like(train_labels_hira), train_labels_hira))
train_labels_kata = np.column_stack((np.ones_like(train_labels_kata), train_labels_kata))

train_images = np.concatenate([train_images_hira, train_images_kata])
train_labels = np.concatenate([train_labels_hira, train_labels_kata])

dataset = Dataset(train_images, train_labels)

batch_size = 64

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = KanaModel(in_channels=1, d_model=64, out_base=labels.n_base, out_dakuten=3)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, mode='min')

n_epoch = 30
i = 0
total_batches = len(dataloader) * n_epoch

times = []

for epoch in range(n_epoch):
    losses = []
    for batch_x, (y_base, y_diacritic) in dataloader:
        (x_script, x_image) = batch_x

        if x_image.shape[0] < batch_size:
            continue

        start = time.time()

        x_image = x_image.unsqueeze(1)

        # flatten the image but keep the batch
        out_base, out_dakuten = model(x_script, x_image)

        loss_base = loss_fn(out_base, y_base)
        loss_dakuten = loss_fn(out_dakuten, y_diacritic)

        loss = loss_base + loss_dakuten

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # timing
        end = time.time()
        times.append(end - start)
        time_for_batch = np.mean(times)

        losses.append(loss.item())

        if i % 10 == 0:
            est_remaining = int((total_batches - i) * time_for_batch)
            print(f"\rProgress: {i}/{total_batches} "
                  f"({(i * 100 / total_batches):.2f}%) | "
                  f"ETA: {est_remaining} sec | Loss: {loss.item():.8f}",
                  end="", flush=True)
        i += 1

    scheduler.step(np.mean(losses))

    print("finished epoch", epoch)
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"models/KanaNet_e{epoch}.pth")