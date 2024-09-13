import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # get original tuple from base class
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        # add the image path to the tuple
        path, _ = self.samples[index]
        return original_tuple + (path,)

def create_data_loaders(data_dir, batch_size=32, image_size=(224, 224)):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = CustomImageFolder(root=train_dir, transform=train_transform)
    val_dataset = CustomImageFolder(root=val_dir, transform=val_test_transform)
    test_dataset = CustomImageFolder(root=test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader