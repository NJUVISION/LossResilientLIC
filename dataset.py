import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder

def get_dataloader(config):
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(config['patchsize']), transforms.ToTensor()]
    )
    train_dataset = ImageFolder(config['trainset'], split="train", transform=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batchsize'],
        num_workers=config['worker_num'],
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )


    test_transforms = transforms.Compose(
        [transforms.CenterCrop(config['patchsize']), transforms.ToTensor()]
    )
    test_dataset = ImageFolder(config['trainset'], split="test", transform=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batchsize_test'],
        num_workers=config['worker_num'],
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, test_dataloader
