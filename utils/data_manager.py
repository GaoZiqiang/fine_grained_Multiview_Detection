import glob
import os.path as osp

import torch
import torchvision

def get_data_loader(dir_path,transform,batch_size):
    # dir_path = './input'
    img_paths = glob.glob(dir_path)
    testloaders = []
    i = 0
    for img_path in img_paths:
        if i > 1:
            break
        i += 1
        testset = torchvision.datasets.ImageFolder(root=img_path,
                                                    transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloaders.append(testloader)

    return testloaders

def get_data_loader_8(dir_path,transform,batch_size):
    # dir_path = './input'
    img_paths = glob.glob(dir_path)
    testloaders = []
    i = 0
    for img_path in img_paths:
        if i > 7:
            break
        i += 1
        testset = torchvision.datasets.ImageFolder(root=img_path,
                                                    transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloaders.append(testloader)

    return testloaders
