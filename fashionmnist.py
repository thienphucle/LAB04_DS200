import os
import pickle
from torchvision import datasets, transforms

def save_batches(dataset, name_prefix, output_folder, batch_size=128):
    os.makedirs(output_folder, exist_ok=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, (images, labels) in enumerate(data_loader):
        images = images.numpy().reshape(images.shape[0], -1)
        labels = labels.numpy().tolist()
        batch_data = {
            b'data': images,
            b'labels': labels
        }

        with open(os.path.join(output_folder, f"{name_prefix}_batch_{i+1}.pkl"), "wb") as f:
            pickle.dump(batch_data, f)
    print(f"Saved {i+1} batches to {output_folder}")

if __name__ == "__main__":
    import torch

    output_dir = "data/fashion_mnist"
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.FashionMNIST(root=".", train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root=".", train=False, transform=transform, download=True)

    save_batches(train_dataset, "train", output_dir)
    save_batches(test_dataset, "test", output_dir)
