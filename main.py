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
# 48 x 48 (2304)
train_images_hira = np.load("hiragana_train_images.npz")['arr_0']
train_labels_hira = np.load("hiragana_train_labels.npz")['arr_0']
train_images_kata = np.load("katakana_train_images.npz")['arr_0']
train_labels_kata = np.load("katakana_train_labels.npz")['arr_0']

train_labels_kata = np.array([label + len(labels.hiragana) for label in train_labels_kata])

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
    for batch_x, (y_script, y_base, y_diacritic) in dataloader:
        if batch_x.shape[0] < batch_size:
            continue

        start = time.time()

        batch_x = batch_x.unsqueeze(1)

        # flatten the image but keep the batch
        out_script, out_base, out_dakuten = model(batch_x)

        loss_script = loss_fn(out_script, y_script)
        loss_base = loss_fn(out_base, y_base)
        loss_dakuten = loss_fn(out_dakuten, y_diacritic)

        loss = 2* loss_script + loss_base + 0.5 * loss_dakuten

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


test_images_hira = np.load("hiragana_test_images.npz")['arr_0']
test_images_kata = np.load("katakana_test_images.npz")['arr_0']
test_labels_hira = np.load("hiragana_test_labels.npz")['arr_0']
test_labels_kata = np.load("katakana_test_labels.npz")['arr_0']

test_labels_kata = np.array([label + len(labels.hiragana) for label in test_labels_kata])
test_images = np.concatenate([test_images_hira, test_images_kata])
test_labels = np.concatenate([test_labels_hira, test_labels_kata])

test_dataset = Dataset(test_images, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

correct = 0
total = len(test_dataset)

for batch_x, batch_y in test_dataloader:
    prediction = model(batch_x)
    if prediction.argmax() == batch_y.argmax():
        print("guessed correctly #", batch_y.argmax())
        correct += 1
    else:
        print("guessed incorrectly #", batch_y.argmax())

print(f"Accuracy: {100 * correct / total}")
