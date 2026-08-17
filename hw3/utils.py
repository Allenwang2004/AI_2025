# utils.py
from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name

def load_train_dataset(path: str='data/train/') -> Tuple[List, List]:
    images = []
    labels = []
    label_map = {"elephant": 0, "jaguar": 1, "lion": 2, "parrot": 3, "penguin": 4}

    for animal in os.listdir(path):
        animal_path = os.path.join(path, animal)
        if os.path.isdir(animal_path):
            for img_file in os.listdir(animal_path):
                images.append(os.path.join(animal_path, img_file))
                labels.append(label_map[animal])
    return images, labels

def load_test_dataset(path: str='data/test/') -> List:
    images = []
    for img_file in os.listdir(path):
        images.append(os.path.join(path, img_file))
    return images

def plot(train_losses: List, val_losses: List):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()
    print("Save the plot to 'loss.png'")