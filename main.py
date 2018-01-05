# ver 2.0
# lr = 0.002, epoch=25, useBN, batchSize=8, Adam, BMSE, layer = 3  Dice = 0.4481 MSELoss = 1.14E-05
'''
######### Do not use Korean or error in my computer################
'''

# work done
# - add paser for adjusting the number of dirac layer

##############what we have to do####
# 1. determine learning rate star point  -> LSJ
# 2. learning rate decay
# 3. loss function (dice + cross + regularlization) -> LSW : Cheer up!
# 4. launch a google cloud instance for latest
# 5. csv submission for val accuracy
# 6. change the model structure (depth, width...)
# 7. data augmentation -> BSH



from dataset import *
from model import Net

import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

from torch.autograd import Variable
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('dataroot', help='path to dataset of kaggle ultrasound nerve segmentation')
# parser.add_argument('dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='number of epoch to start')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--useBN', action='store_true', help='enalbes batch normalization')
parser.add_argument('--output_name', default='checkpoint___.tar', type=str, help='output checkpoint filename')
parser.add_argument('--layer', default='3', type=int, help='number of layer')

args = parser.parse_args()
print(args)


############## dataset processing
dataset = kaggle2016nerve(args.dataroot)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                           num_workers=args.workers, shuffle=True)
dataset = kaggle2016nerve(args.dataroot, False)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         num_workers=args.workers, shuffle=False)


############## create model
model = Net(args.useBN)
if args.cuda:
    model.cuda()
    cudnn.benchmark = True


    ############## resumeif args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))

        if args.cuda == False:
            checkpoint = torch.load(args.resume, map_location={'cuda:0': 'cpu'})

        args.start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, loss {})"
              .format(checkpoint['epoch'], checkpoint['loss']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


def save_checkpoint(state, filename=args.output_name):
    torch.save(state, filename)

# Learning Rate Decay
def adjust_learning_rate(optimizer, epoch, n):
    """Sets the learning rate to the initial LR decayed by 10 every n epochs"""
    lr = args.lr * (0.1 ** (epoch // n))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

'''
def dice_coff(pred, truth):
    pred = pred[0, 0, :, :]
    truth = truth[0, 0, :, :]
    sum_pred = 0
    sum_truth = 0
    overlap = 0
    shape = pred.shape
    print('shape[0] = {}, shape[1] = {}'.format(shape[0], shape[1]))
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if (pred[i][j] > 0.5):
                pred[i][j] = 1
                sum_pred += 1
            else:
                pred[i][j] = 0

            if (truth[i][j] > 0.5):
                truth[i][j] = 1
                sum_truth += 1
            else:
                truth[i][j] = 0
            overlap += pred[i][j] * truth[i][j]    if (sum_truth == 0):  # if an image does not have an annotation == 0
        if (sum_pred == 0):
            return 1
        else:
            return 0

    if (sum_truth == 0):  # if an image does not have an annotation == 0
        if (sum_pred == 0):
            return 1
        else:
            return 0
    else:
        return (2 * overlap / (sum_pred + sum_truth))
'''

def dice_coff(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = im1[0, 0, :, :]
    im2 = im2[0, 0, :, :]
    im1 = np.asarray(im1) > 0.5
    im2 = np.asarray(im2) > 0.5 
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

############## trainin
optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
model.train()
'''
0.406653249674
0.0516689529035
0.155193030407
0.277971574365
0.662277275602
0.454754182004
0.301195568718
0.364192139738
0.193085287122
0.228934768217
0.387603985739
Dice Cofficient of Train: 0.3167, Epoch: 0
Dice Cofficient of Val: 0.3997, Epoch: 0

Dice coefficient 맞는 듯 아래가 문제..
우리 카페감 전화하셈

 '''

# training data is now split into train/validation data (80%/20%)
def showImg(img, binary=True, fName=''):
    """
    show image from given numpy image
    """
    img = img[0, 0, :, :]

    if binary:
        img = img > 0.5

    img = Image.fromarray(np.uint8(img * 255), mode='L')

    if fName:
        img.save('assets/' + fName + '.png')
    else:
        img.show()


def train(epoch):
    """
    training
    """
    loss_fn = nn.MSELoss()  # Dice
    if args.cuda:
        loss_fn = loss_fn.cuda()

    loss_sum = 0

    for i, (x, y) in enumerate(train_loader):
        x, y_true = Variable(x), Variable(y)
        if args.cuda:
            x = x.cuda()
            y_true = y_true.cuda()

        for ii in range(1):
            y_pred = model(x)

            loss = loss_fn(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            loss_sum += loss.data[0]

            optimizer.step()

        if i % 5 == 0:
            print('batch no.: {}, loss: {}'.format(i, loss.data[0]))
    
    print('epoch: {}, epoch loss: {}'.format(epoch, loss.data[0] / len(train_loader)))
    f3.write(str(loss.data[0] / len(train_loader)))
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'loss': loss.data[0] / len(train_loader)
    })


f1 = open("trainDice.txt", 'w')
f2 = open("valDice.txt", 'w')
f3 = open("loss.txt", 'w')


for epoch in range(args.niter):
    Sum_val_dice_coff = 0
    Sum_train_dice_coff = 0
    train(epoch)
    #### save generated img to compare with test img
    for i, (x, y) in enumerate(train_loader):
        if i >= 11:
            break
        y_pred = model(Variable(x.cuda()))
        print(dice_coff(y_pred.data.cpu().numpy(), y.numpy()))
        Sum_train_dice_coff += dice_coff(y_pred.data.cpu().numpy(), y.numpy())
        NumOfVal = i + 1
        showImg(x.cpu().numpy(), binary=False, fName='ori_' + str(i) + '_' + str(epoch))
        showImg(y_pred.data.cpu().numpy(), binary=False, fName='pred_' + str(i) + '_' + str(epoch))
        showImg(y_pred.data.cpu().numpy(), binary=True, fName='pred_' + str(i) + '_' + str(epoch) + 'b')
        showImg(y.numpy(), fName='gt_' + str(i) + '_' + str(epoch))
    avgOftrain = Sum_train_dice_coff / NumOfVal
    print("Dice Cofficient of Train: {:0.4f}, Epoch: {}".format(avgOftrain, epoch))
    f1.write(str(avgOftrain))

    for i, (x, y) in enumerate(val_loader):
        if (i >= 11):
            break
        y_pred = model(Variable(x.cuda()))
        Sum_val_dice_coff += dice_coff(y_pred.data.cpu().numpy(), y.numpy())
        NumOfVal = i + 1

    avgOfVal = Sum_val_dice_coff / NumOfVal
    print("Dice Cofficient of Val: {:0.4f}, Epoch: {}".format(avgOfVal, epoch))
    f2.write(str(avgOfVal))
    ### learning rate decay
    adjust_learning_rate(optimizer, epoch, 5)


f1.close()
f2.close()
f3.close()
model.eval()

Sum_dice_coff = 0
for i, (x, y) in enumerate(val_loader):
    y_pred = model(Variable(x.cuda()))
    coff = dice_coff(y_pred.data.cpu().numpy(), y.numpy())
    print("Dice Coefficient of Image{}: {:0.4f}".format(i+1, coff))
    Sum_dice_coff += coff
    NumOfVal = i + 1

    print("Average Dice Coefficient: {:0.4f}".format(Sum_dice_coff / NumOfVal))

train_loader.batch_size = 1


# def test():
#    dataset = kaggle2016nerve(args.dataroot, False)
#    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
#                                               num_workers=args.workers, shuffle=True)ye

