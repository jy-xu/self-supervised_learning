from torchvision import datasets
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

class CIFAR10_SimCLR(datasets.CIFAR10):
    def __init__(self, transform_no = None, *args, **kwargs):
        super(CIFAR10_SimCLR, self).__init__(*args, **kwargs)
        self.transform_no = transform_no

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            xi = self.transform(img)
            xj = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.transform_no is not None:
            x = self.transform_no(img)
            return xi, xj, x, target
        else:
            return xi, xj, target


def get_dataloader(transform_train, transform_test, transform_no=None, data_dir='./data', batch_size=256, perc=1):
    trainset = CIFAR10_SimCLR(root=data_dir, train=True, download=True, transform=transform_train, transform_no=transform_no)
    valset = CIFAR10_SimCLR(root=data_dir, train=False, download=True, transform=transform_test, transform_no=transform_no)
    
    if perc == 1:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=perc, random_state=0)
        _, train_index = next(sss.split(trainset.data, trainset.targets))
        labeled_train = Subset(trainset, train_index)
        trainloader = DataLoader(labeled_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        _, val_index = next(sss.split(valset.data, valset.targets))
        labeled_val = Subset(valset, val_index)
        valloader = DataLoader(labeled_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
    return trainloader, valloader