from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import os
import numpy as np

class SavingNatureDataset(Dataset):
    def __init__(self, root, transform=None, transform_no=None):
        self.root = root
        self.transform = transform
        self.transform_no = transform_no
        self.target_transform = None
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.labels = sorted(os.listdir(os.path.join(root, "labels")))
        self.targets, self.data = self._getdatatargets()
    
    def __getitem__(self, index):
        # load images and bounding boxes
        img_path = os.path.join(self.root, "images", self.imgs[index])
        label_path = os.path.join(self.root, "labels", self.labels[index])

        img = Image.open(img_path).convert("RGB")
        #img = img.resize((32, 32))
        if os.stat(label_path).st_size != 0: #check if file is empty
            target = np.genfromtxt(label_path, dtype=int, usecols=[4]) -1
        else:
            target = 0

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
    
    def __len__(self):
         return len(self.imgs)
    
    def _getdatatargets(self):
        targets = []
        data = []
        for i in len(self.imgs):
            image, _, target = self.__getitem__(i)
            targets.append(target)
            data.append(image)
        return torch.cat(targets, dim=1), torch.cat(data, dim=1)
        
def get_dataloader(transform_train, transform_test, transform_no=None, data_dir='./data', batch_size=256, perc=1):
    trainset = SavingNatureDataset(root=data_dir + "/train", transform=transform_train, transform_no=transform_no)
    valset = SavingNatureDataset(root=data_dir + "/val", transform=transform_test, transform_no=transform_no)
    
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