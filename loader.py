
import os
import os.path as osp
from astropy.io import fits
import torch.utils.data as data

IMG_EXTENSIONS = ['fits']
EXTTYPE = 'EXTTYPE'
IMAGE = 'IMAGE'


def make_dataset(dir):
    images = []
    for path, _, files in os.walk(dir):
        for filename in files:
            print(filename)
            name, ext = osp.splitext(filename)
            print(ext)
            if ext in IMG_EXTENSIONS:
                filename = osp.join(path, filename)
                images.append(filename)
    return images


def astropy_loader(path):
    hdulist = fits.open(path)
    img = None
    for hdu in hdulist:
        if EXTTYPE in hdu.header():
            if hdu.header[EXTTYPE] == IMAGE:
                img = hdu.data
    if img is None:
        err = "The file {0} does not contain any hdu image"
        raise RuntimeError(err.format(path))


class TransientObjectLoader(data.Dataset):
    def __init__(self, root, transform=None, train=True,
                 loader=astropy_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            ext = ','.join(IMG_EXTENSIONS)
            raise RuntimeError("The path {0} does not "
                               "contain any images of "
                               "extension {1}".format(root, ext))
        self.root = root
        # self.imgs = imgs
        if train:
            self.imgs = imgs[:60000]
        else:
            self.imgs = imgs[60000:]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
