import torch.utils.data as data
import torch

from scipy.ndimage import imread
import os
import os.path
import glob

import numpy as np

from torchvision import transforms

def make_dataset(root, type='train'):

  dataset = []

  if type == 'train':
    dir = os.path.join(root, 'train')
    for fGT in glob.glob(os.path.join(dir, '*_mask.tif')):
      fName = os.path.basename(fGT)
      fImg = fName[:-9] + '.tif'
      dataset.append( [os.path.join(dir, fImg), os.path.join(dir, fName)] )

  elif type == 'test' :
    dir = os.path.join(root, 'test')
    for fGT in glob.glob(os.path.join(dir, '*_mask.tif')):
      fName = os.path.basename(fGT)
      fImg = fName[:-9] + '.tif'
      dataset.append( [os.path.join(dir, fImg), os.path.join(dir, fName)] )

  elif type == 'val' :
    dir = os.path.join(root, 'val')
    for fGT in glob.glob(os.path.join(dir, '*.tif')):
      fName = os.path.basename(fGT)
#      fImg = fName + '.tif'
#      dataset.append([os.path.join(dir, fImg), os.path.join(dir, fName)])
      dataset.append( [os.path.join(dir, fName), os.path.join(dir, fName)] )
  return dataset

class kaggle2016nerve(data.Dataset):
  """
  Read dataset of kaggle ultrasound nerve segmentation dataset
  https://www.kaggle.com/c/ultrasound-nerve-segmentation
  """

  def __init__(self, root, transform=None, type='train'):
    self.type = type

    # we cropped the image
    self.nRow = 400
    self.nCol = 560

    if self.type == 'train':
      self.train_set_path = make_dataset(root, 'train')
    elif self.type == 'test':
      self.test_set_path = make_dataset(root, 'test')
    elif self.type == 'val':
      self.val_set_path = make_dataset(root, 'val')

  def __getitem__(self, idx):
    if self.type == 'train':
      img_path, gt_path = self.train_set_path[idx]

      img = imread(img_path)
      img = img[0:self.nRow, 0:self.nCol]
      img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
      img = (img - img.min()) / (img.max() - img.min())
      img = torch.from_numpy(img).float()

      gt = imread(gt_path)[0:self.nRow, 0:self.nCol]
      gt = np.atleast_3d(gt).transpose(2, 0, 1)
      gt = gt / 255.0
      gt = torch.from_numpy(gt).float()

      return img, gt

    elif self.type == 'test':
      img_path, gt_path = self.test_set_path[idx]

      img = imread(img_path)
      img = img[0:self.nRow, 0:self.nCol]
      img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
      img = (img - img.min()) / (img.max() - img.min())
      img = torch.from_numpy(img).float()

      gt = imread(gt_path)[0:self.nRow, 0:self.nCol]
      gt = np.atleast_3d(gt).transpose(2, 0, 1)
      gt = gt / 255.0
      gt = torch.from_numpy(gt).float()

      return img, gt

    elif self.type == 'val':
      img_path, gt_path = self.val_set_path[idx]

      img = imread(img_path)
      img = img[0:self.nRow, 0:self.nCol]
      img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
      img = (img - img.min()) / (img.max() - img.min())
      img = torch.from_numpy(img).float()

      gt = imread(gt_path)[0:self.nRow, 0:self.nCol]
      gt = np.atleast_3d(gt).transpose(2, 0, 1)
      gt = gt / 255.0
      gt = torch.from_numpy(gt).float()

      return img, gt

  def __len__(self):
    if self.type == 'train':
      return 5635-1127
    elif self.type == 'test' :
      return 1127
    elif self.type == 'val':
      return 5508
