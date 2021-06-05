from pathlib import Path

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CropFace:
    def __init__(self, detector, expand_rate=.4):
        self.detector = detector
        self.expand_rate = expand_rate

    def __call__(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rect = self.detector(gray, 1)
        if len(rect) < 1:
            return image
        rect = rect[0]

        w = rect.right() - rect.left()
        h = rect.bottom() - rect.top()

        left = max((0, rect.left() - int(w*self.expand_rate)))
        right = min((image.shape[0], rect.right() + int(w*self.expand_rate)))
        top = max((0, rect.top() - int(h*self.expand_rate)))
        bottom = min((image.shape[1], rect.bottom() + int(h*self.expand_rate)))

        return image[left: right, top: bottom]


class HairDataset(Dataset):
    def __init__(self, long_path, short_path, detector):
        long_entries = list(sorted(Path(long_path).glob('*.jpg')))
        short_entries = list(sorted(Path(short_path).glob('*.jpg')))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            CropFace(detector),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2),
            transforms.ToTensor(),
            normalize])
        self.entries = long_entries + short_entries
        self.long_count = len(long_entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        path = self.entries[item]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        label = int(item < self.long_count)
        return self.transform(img), label


class ValDataset(Dataset):
    def __init__(self, path, detector):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                CropFace(detector),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize
            ])
        self.entries = list(sorted(Path(path).glob('*.jpg')))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        path = self.entries[item]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return self.transform(img)
