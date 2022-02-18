from sklearn.utils import shuffle
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# import codecs
# import errno
# import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
# import random
import torch
from torch import nn
#from torch import optim
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
# from tqdm import tqdm
import importlib.machinery
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
import argparse
from torch.autograd import Variable
from sklearn.utils import class_weight
import json
import math
import collections
from skimage import img_as_float, transform, exposure
from skimage.transform import resize
import random
import cv2
import pandas as pd
import re


    ##
    ##  UTILITY FUNCTIONS
    ##

process_results = importlib.machinery.SourceFileLoader('process_results','./ProcessResults/process_results.py').load_module()

SEED = 42

def flatten(lst):
    ## from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    return eval('[' + str(lst).replace('[', '').replace(']', '') + ']')


def load_dataset(json_infile, test_prop, data_dir, ordered_split=False, train_only=False):
    # print("break")
    with open(data_dir + json_infile, 'r') as fp:
        in_dict = json.load(fp)

    if 'SU2bae8dc' in list(in_dict.keys()):
        in_dict.pop('SU2bae8dc', None)

    if train_only:
        train_out = in_dict

        return train_out

    else:
        pt_ids = list(in_dict.keys())
        BL_dates = []
        for my_key in pt_ids:
            if 'BL_date' in in_dict[my_key].keys():
                BL_dates.extend([in_dict[my_key]['BL_date']])
            else:
                BL_dates.extend(['2021-01-01'])

        if ordered_split:
            sorted_pt_BL_dates = sorted(zip(pt_ids, BL_dates))
            test_n = round(len(pt_ids) * test_prop)
            train_out = dict()
            test_out = dict()

            for i in range(len(pt_ids) - test_n):
                study_id, _ = sorted_pt_BL_dates[i]
                train_out[study_id] = in_dict[study_id]

            for i in range(len(pt_ids) - test_n + 1, len(pt_ids)):
                study_id, _ = sorted_pt_BL_dates[i]
                test_out[study_id] = in_dict[study_id]

        else:
            shuf_pt_id, shuf_BL_dates = shuffle(list(pt_ids), BL_dates, random_state=42)
            test_n = round(len(shuf_pt_id)*test_prop)
            train_out = dict()
            test_out = dict()

            for i in range(test_n):
                study_id = shuf_pt_id[i]
                test_out[study_id] = in_dict[study_id]

            for i in range(test_n+1, len(shuf_pt_id)):
                study_id = shuf_pt_id[i]
                train_out[study_id] = in_dict[study_id]

        return train_out, test_out


def load_test_dataset(json_infile, data_dir):

    with open(data_dir + json_infile, 'r') as fp:
        in_dict = json.load(fp)

    if 'SU2bae8dc' in list(in_dict.keys()):
        in_dict.pop('SU2bae8dc', None)

    return in_dict


# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

    ##
    ## NN
    ##

## alternative model w/ vs w/o covariates
class SiamNet_v2(nn.Module):
    def __init__(self, classes=2, num_inputs=2, cov_layers=True, one_hot=True):
        super(SiamNet_v2, self).__init__()

        self.cov_layers = cov_layers
        self.one_hot = one_hot

        # print("LL DIM: " + str(self.output_dim))
        self.num_inputs = num_inputs

        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu3_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu4_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        # *************************** changed layers *********************** #
        self.fc6 = nn.Sequential()
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1))
        self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=1))
        self.fc6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))

        self.fc6b = nn.Sequential()
        self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(256))
        self.fc6b.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6b.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6c = nn.Sequential()
        # self.fc6c.add_module('fc7', nn.Linear(256*2*2, 512))
        self.fc6c.add_module('fc7', nn.Linear(256 * 3 * 3, 512))
        self.fc6c.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc6c.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear(self.num_inputs * 512, 256))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc7_new.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc8', nn.Linear(256, classes))

        if self.cov_layers:
            if self.one_hot:
                self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

                self.add_covs1 = nn.Sequential()
                self.add_covs1.add_module('fc9_onehot', nn.Linear(classes + 6, classes))

            else:
                self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

                self.add_covs1 = nn.Sequential()
                self.add_covs1.add_module('fc9', nn.Linear(classes + 2, classes))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x, get_embeddings=False):

        if self.cov_layers:
            in_dict = x
            x = in_dict['img']

        # x = x.unsqueeze(0)
        if self.num_inputs == 1:
            x = x.unsqueeze(1)
        #   B, C, H = x.size()
        # else:
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            curr_x = curr_x.expand(-1, 3, -1, -1) ## expanding 1 channel to 3 duplicate channels
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))
            z = self.conv(input)
            z = self.fc6(z) ## convolution
            z = self.fc6b(z) ## convolution
            z = z.view([B, 1, -1])
            z = self.fc6c(z) ## fully connected layer
            ### LAUREN CHECK THIS -- shouldn't need to .view, no?
            # z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)

        if get_embeddings:
            embed = x.view(B, -1)

        # x = torch.sum(x, 1)
        x = self.fc7_new(x.view(B, -1))
        pred = self.classifier_new(x)

        if self.cov_layers:
            age = in_dict['Age_wks'].type(torch.FloatTensor).to(device).view(B, 1)
            # print("Age: ")
            # print(age)
            side = in_dict['Side_L'].type(torch.FloatTensor).to(device).view(B, 1)
            # print("Side: ")
            # print(side)

            if self.one_hot:
                one_hot = torch.transpose(torch.cat(in_dict['one_hot']).view(4, B).type(torch.FloatTensor).to(device), 0, 1)
                mid_in = torch.cat((pred, age, side, one_hot), 1)

            else:
                mid_in = torch.cat((pred, age, side), 1)

            pred = self.add_covs1(mid_in)

        if get_embeddings:
            return pred, embed
        else:
            return pred

## orignal model being used w/ vs w/o covariates
class SiamNet(nn.Module):
    def __init__(self, classes=2, num_inputs=2, cov_layers=True, one_hot=True):
        super(SiamNet, self).__init__()

        self.cov_layers = cov_layers
        self.one_hot = one_hot

        # print("LL DIM: " + str(self.output_dim))
        self.num_inputs = num_inputs

        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu3_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu4_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        # *************************** changed layers *********************** #
        self.fc6 = nn.Sequential()
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1))
        self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        # self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=1))
        self.fc6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))

        self.fc6b = nn.Sequential()
        self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(256))
        self.fc6b.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6b.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6c = nn.Sequential()
        # self.fc6c.add_module('fc7', nn.Linear(256*2*2, 512))
        self.fc6c.add_module('fc7', nn.Linear(256 * 3 * 3, 512))
        self.fc6c.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc6c.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear(self.num_inputs * 512, 256))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc7_new.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc8', nn.Linear(256, classes))

        if self.cov_layers:
            if self.one_hot:
                self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

                self.add_covs1 = nn.Sequential()
                self.add_covs1.add_module('fc9_onehot', nn.Linear(classes + 6, classes + 126))
                self.add_covs1.add_module('relu9', nn.ReLU(inplace=True))

                self.add_covs2 = nn.Sequential()
                self.add_covs2.add_module('fc10', nn.Linear(classes + 126, classes))
            else:
                self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

                self.add_covs1 = nn.Sequential()
                self.add_covs1.add_module('fc9', nn.Linear(classes + 2, classes + 126))
                self.add_covs1.add_module('relu9', nn.ReLU(inplace=True))

                self.add_covs2 = nn.Sequential()
                self.add_covs2.add_module('fc10', nn.Linear(classes + 126, classes))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x, get_embeddings=False):

        if self.cov_layers:
            in_dict = x
            x = in_dict['img']

        # x = x.unsqueeze(0)
        if self.num_inputs == 1:
            x = x.unsqueeze(1)
        #   B, C, H = x.size()
        # else:
        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            if self.num_inputs == 1:
                curr_x = torch.unsqueeze(x[i], 1)
            else:
                curr_x = torch.unsqueeze(x[i], 1)
            # if self.num_inputs == 1:
            #   curr_x = curr_x.expand(-1, 3, -1)
            # else:
            curr_x = curr_x.expand(-1, 3, -1, -1) ## expanding 1 channel to 3 duplicate channels
            if torch.cuda.is_available():
                input = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input = torch.FloatTensor(curr_x.to(device))
            z = self.conv(input)
            z = self.fc6(z) ## convolution
            z = self.fc6b(z) ## convolution
            z = z.view([B, 1, -1])
            z = self.fc6c(z) ## fully connected layer
            ### LAUREN CHECK THIS -- shouldn't need to .view, no?
            # z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)

        if get_embeddings:
            embed = x.view(B, -1)

        # x = torch.sum(x, 1)
        x = self.fc7_new(x.view(B, -1))
        pred = self.classifier_new(x)

        if self.cov_layers:
            age = in_dict['Age_wks'].type(torch.FloatTensor).to(device).view(B, 1)
            # print("Age: ")
            # print(age)
            side = in_dict['Side_L'].type(torch.FloatTensor).to(device).view(B, 1)
            # print("Side: ")
            # print(side)

            if self.one_hot:
                one_hot = torch.transpose(torch.cat(in_dict['one_hot']).view(4, B).type(torch.FloatTensor).to(device), 0, 1)
                mid_in = torch.cat((pred, age, side, one_hot), 1)

            else:
                mid_in = torch.cat((pred, age, side), 1)

            x = self.add_covs1(mid_in)
            pred = self.add_covs2(x)

        if get_embeddings:
            return pred, embed
        else:
            return pred


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1, padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)

            div = div.mul(self.alpha).add(1.0).pow(self.beta)

        x = x.div(div)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)

    ##
    ## DATA LOADER
    ##


def pad_img_le(image_in, dim=256, random_pad = False):
    """

    :param image_in: input image
    :param dim: desired dimensions of padded image (should be bigger or as big as all input images)
    :return: padded image

    """

    im_shape = image_in.shape

    while im_shape[0] > dim or im_shape[1] > dim:
        image_in = resize(image_in, ((im_shape[0]*4)//5, (im_shape[1]*4)//5), anti_aliasing=True)
        im_shape = image_in.shape

    # print(im_shape)
    if random_pad:

        if random.random() >= 0.5:
            image_in = cv2.flip(image_in, 1)

        rand_h = np.random.uniform(0, 1, 1)
        rand_v = np.random.uniform(0, 1, 1)
        right = math.floor((dim - im_shape[1])*rand_h)
        left = math.ceil((dim - im_shape[1])*(1-rand_h))
        bottom = math.floor((dim - im_shape[0])*rand_v)
        top = math.ceil((dim - im_shape[0])*(1-rand_v))
    else:
        right = math.floor((dim - im_shape[1])/2)
        left = math.ceil((dim - im_shape[1])/2)
        bottom = math.floor((dim - im_shape[0])/2)
        top = math.ceil((dim - im_shape[0])/2)

    image_out = cv2.copyMakeBorder(image_in, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return image_out


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    from fviktor here: https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def set_contrast(image, contrast=1):
    """
    Set contrast in image

    :param image: input image
    :param contrast: contrast type
    :return: image with revised contrast
    """

    if contrast == 0:
        out_img = image
    elif contrast == 1:
        out_img = exposure.equalize_hist(image)
    elif contrast == 2:
        out_img = exposure.equalize_adapthist(image)
    elif contrast == 3:
        out_img = exposure.rescale_intensity(image)

    return out_img


# CROP IMAGES #
def crop_image(image, random_crop=False):
    # IMAGE_PARAMS = {0: [2.1, 4, 4],
    #                 # 1: [2.5, 3.2, 3.2],
    #                 # 2: [1.7, 4.5, 4.5] # [2.5, 3.2, 3.2] # [1.7, 4.5, 4.5]
    #                 }
    width = image.shape[1]
    height = image.shape[0]

    if random_crop:
        par1 = random.uniform(1.5, 2.8)
        par2 = random.uniform(3, 4.5)
        par3 = random.uniform(3, 4.5)


    else:
        par1 = 2.1
        par2 = 4
        par3 = 4

    new_dim = int(width // par1)  # final image will be square of dim width/2 * width/2

    start_col = int(width // par2)  # column (width position) to start from
    start_row = int(height // par3)  # row (height position) to start from

    cropped_image = image[start_row:start_row+new_dim, start_col:start_col+new_dim]

    return cropped_image


def special_ST_preprocessing(img_file, output_dim=256):

    # print("stop here")

    if "preprocessed" in img_file:
        my_img = process_input_image(img_file)

    else:
        image = np.array(Image.open(img_file).convert('L'))
        image_grey = img_as_float(image)

        # if crop:
        #     cropped_img = crop_image(image_grey)
        # else:
        #     cropped_img = image_grey
        cropped_img = image_grey

        resized_img = transform.resize(cropped_img, output_shape=(output_dim, output_dim))
        my_img = set_contrast(resized_img) ## ultimately add contrast variable
        img_name = img_file.split('/')[len(img_file.split('/')) - 1]
        img_folder = "/".join(img_file.split('/')[:-2])
        out_img_filename = img_folder + "/Preprocessed/" + img_name.split('.')[0] + "-preprocessed.png"
        if not os.path.exists(img_folder + "/Preprocessed/"):
            os.makedirs(img_folder + "/Preprocessed/")

        # out_img_filename = 'C:/Users/lauren erdman/Desktop/kidney_img/img_debugging/' + img_name.split('.')[0] + "-preprocessed.png"
        Image.fromarray(my_img*255).convert('RGB').save(out_img_filename)

        my_img = np.array(Image.open(out_img_filename).convert('L'))
        ## Add random cropping/padding

        # print("pause")
        ## DEBUG FROM HERE

    return my_img


def process_input_image(img_file, crop=False, random_crop=False):
    """
    Processes image: crop, convert to greyscale and resize

    :param image: image
    :return: formatted image
    """
    fit_img = np.array(Image.open(img_file).convert('L'))
    if fit_img.shape[0] != 256 or fit_img.shape[1] != 256:
        print(img_file)
        fit_img = special_ST_preprocessing(img_file)

    ## add random cropping/padding

    # if random_crop:
    #     # image = set_contrast(crop_image(image)) ## adding contrast here is not helping
    #     image = crop_image(image, random_crop=False)
    #     fit_img = pad_img_le(np.array(image), random_pad=True)
    #
    #
    # else:
    #     if crop:
    #         # image = set_contrast(crop_image(image)) ## adding contrast here is not helping
    #         image = crop_image(image)
    #         fit_img = pad_img_le(np.array(image))
    #
    #     else:
    #         # print("standard preprocessing, no cropping")
    #         # image = np.array(Image.open(img_file).convert('L'))
    #         # fit_img = pad_img_le(np.array(image))
    #         fit_img = np.array(Image.open(img_file).convert('L'))
    #
    #         if fit_img.shape[0] != 256 or fit_img.shape[1] != 256:
    #             print(img_file)
    #             fit_img = special_ST_preprocessing(img_file)

    return fit_img


def get_images(in_dict, data_dir, crop=False, random_crop=False, update=False, update_num=None, silent_trial=False,
               ablate_age=False):
    img_dict = dict()
    label_dict = dict()
    cov_dict = dict()

    print("break")

    for study_id in in_dict.keys():

        try:
            try:
                one_hot = in_dict[study_id]['one_hot']
            except KeyError:
                if study_id.find("SU") > 0:
                    one_hot = [0, 1, 0, 0]
                elif study_id.find("STID") > 0:
                    one_hot = [1, 0, 0, 0]
                elif study_id.find("ORIG") > 0:
                    one_hot = [1, 0, 0, 0]
                elif study_id.find("CHOP"):
                    one_hot = [0, 0, 0, 1]
                else:
                    one_hot = [0, 0, 1, 0]

            sides = np.setdiff1d(list(in_dict[study_id].keys()), ['BL_date', 'Sex'])
            for side in sides:
                # print(study_id)
                # print(in_dict[study_id][side])

                surgery = None

                if type(in_dict[study_id][side]) == dict:
                    if 'surgery' in list(in_dict[study_id][side].keys()):
                        if type(in_dict[study_id][side]['surgery']) == int:
                            surgery = in_dict[study_id][side]['surgery']
                        elif type(in_dict[study_id][side]['surgery']) == list:
                            s1 = [i for i in in_dict[study_id][side]['surgery'] if type(i) == int]
                            if(len(s1) > 0):
                                surgery = s1[0]
                            else:
                                surgery = None
                        elif type(in_dict[study_id][side]['surgery']) == str:
                            surgery = None
                    else:
                        surgery = None
                else:
                    surgery = None

                us_nums = [my_key for my_key in in_dict[study_id][side].keys() if my_key != 'surgery']
                if len(us_nums) > 0 and surgery is not None:
                    for us_num in us_nums:
                        if in_dict[study_id][side][us_num]['Age_wks'] != "NA" and \
                                set(['sag', 'trv']).issubset(in_dict[study_id][side][us_num].keys()):

                            if update:
                                dict_key = study_id+"_"+side+"_" + us_num
                            else:
                                dict_key = study_id + "_" + side + "_" + us_num + "_" + str(update_num)

                            img_dict[dict_key] = dict()

                            print(data_dir)
                            print(in_dict[study_id][side][us_num]['sag'])
                            print(in_dict[study_id][side][us_num]['trv'])
                            if silent_trial:
                                img_dict[dict_key]['sag'] = process_input_image(data_dir + in_dict[study_id][side][us_num]['sag'], crop=crop, random_crop=random_crop)
                                img_dict[dict_key]['trv'] = process_input_image(data_dir + in_dict[study_id][side][us_num]['trv'], crop=crop, random_crop=random_crop)
                            else:
                                img_dict[dict_key]['sag'] = special_ST_preprocessing(data_dir + in_dict[study_id][side][us_num]['sag'])
                                img_dict[dict_key]['trv'] = special_ST_preprocessing(data_dir + in_dict[study_id][side][us_num]['trv'])
                            # img_dict[dict_key]['sag'] = np.array(Image.open(in_dict[study_id][side][us_num]['sag']).convert('L'))
                            # img_dict[dict_key]['trv'] = np.array(Image.open(in_dict[study_id][side][us_num]['trv']).convert('L'))

                            label_dict[dict_key] = surgery

                            cov_dict[dict_key] = dict()
                            cov_dict[dict_key]["one_hot"] = one_hot

                            try:
                                cov_dict[dict_key]['US_machine'] = in_dict[study_id][side][us_num]['US_machine']
                            except KeyError:
                                cov_dict[dict_key]['US_machine'] = "NA"

                            if in_dict[study_id][side][us_num]['US_machine'] is None:
                                cov_dict[dict_key]['US_machine'] = "NA"

                            if type(in_dict[study_id]['Sex']) == int:
                                cov_dict[dict_key]['Sex'] = in_dict[study_id]['Sex']
                            else:
                                if in_dict[study_id]['Sex'] == "M":
                                    cov_dict[dict_key]['Sex'] = 1
                                else:
                                    cov_dict[dict_key]['Sex'] = 2

                            if ablate_age:
                                cov_dict[dict_key]['Age_wks'] = 0
                            else:
                                cov_dict[dict_key]['Age_wks'] = in_dict[study_id][side][us_num]['Age_wks']

                            # print(label_dict)
                            # print(img_dict[study_id+"_"+side+"_"+us_num])
                            # print(cov_dict[study_id+"_"+side+"_"+us_num])
                            # assert(type(label_dict[dict_key]) is not None)
                            # assert(type(img_dict[dict_key]) is not None)
                            # assert(type(cov_dict[dict_key]['Sex']) is not None)
                            # assert(type(cov_dict[dict_key]['Age_wks']) is not None)

                            assert label_dict[dict_key] is not None
                            assert img_dict[dict_key] is not None
                            assert cov_dict[dict_key]['Sex'] is not None
                            assert cov_dict[dict_key]['Age_wks'] is not None

        except AttributeError:
            continue

    ids = img_dict.keys()

    return img_dict, label_dict, cov_dict, ids


class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, from_full_dict=True, in_dict=None, data_dir=None, image_dict=None, label_dict=None, cov_dict=None, study_ids=None,
                 cov_input=False, rand_crop=False, crop=False, silent_trial=False, ablate_age=False, ablate_side=False):
        print("in_dict")
        print(in_dict)

        if from_full_dict:
            self.image_dict, self.label_dict, self.cov_dict, self.study_ids = get_images(in_dict, data_dir=data_dir,
                                                                                         crop=crop,
                                                                                         random_crop=rand_crop,
                                                                                         update_num=1,
                                                                                         silent_trial=silent_trial,
                                                                                         ablate_age=ablate_age)

            if rand_crop:
                image_dict2, label_dict2, cov_dict2, study_ids2 = get_images(in_dict, data_dir=data_dir, crop=crop,
                                                                             random_crop=rand_crop, update=True,
                                                                             update_num=2,
                                                                             silent_trial=silent_trial,
                                                                             ablate_age=ablate_age)
                self.image_dict.update(image_dict2)
                self.label_dict.update(label_dict2)
                self.cov_dict.update(cov_dict2)

                image_dict3, label_dict3, cov_dict3, study_ids3 = get_images(in_dict, crop=crop,
                                                                             random_crop=rand_crop, update=True,
                                                                             update_num=3,
                                                                             silent_trial=silent_trial,
                                                                             ablate_age=ablate_age)
                self.image_dict.update(image_dict3)
                self.label_dict.update(label_dict3)
                self.cov_dict.update(cov_dict3)

                self.study_ids = image_dict.keys()

        else:
            self.image_dict, self.label_dict, self.cov_dict, self.study_ids = image_dict, label_dict, cov_dict, study_ids

        self.cov_input = cov_input
        self.rand_crop = rand_crop
        self.ablate_side = ablate_side

    def __getitem__(self, index):

        # print(index)

        id = list(self.study_ids)[index]

        # print("ID: ")
        # print(id)

        img, target, cov = self.image_dict[id], self.label_dict[id], self.cov_dict[id]

        # print("Target: ")
        # print(target)

        target_out = torch.tensor(int(target)).to(device).type(torch.DoubleTensor)
        # print("Target converted to tensor.")

        sag_img = torch.tensor(img['sag']).to(device)
        trans_img = torch.tensor(img['trv']).to(device)

        if self.rand_crop:
            trans1 = torchvision.transforms.RandomCrop(200)
            sag_img = torchvision.transforms.functional.pad(trans1(sag_img), padding=28)
            trans_img = torchvision.transforms.functional.pad(trans1(trans_img), padding=28)

        img_out = torch.stack((sag_img, trans_img)).type(torch.FloatTensor)

        # print("Image converted to tensor.")

        if self.cov_input:
            id_split = id.split("_")

            dict_out = cov
            dict_out['img'] = img_out

            if self.ablate_side:
                dict_out['Side_L'] = 0
            else:
                dict_out['Side_L'] = 1 if id_split[1] == 'Left' else 0

            if dict_out['Age_wks'] is None:
                dict_out['Age_wks'] = 36
            elif math.isnan(torch.tensor(dict_out['Age_wks']).type(torch.FloatTensor)):
                dict_out['Age_wks'] = 36

            # print("Side:")
            # print(dict_out['Side_L'])
            # print("Age:")
            # print(dict_out['Age_wks'])
            # print("target:")
            # print(target_out)
            # print("image")
            # print(dict_out["img"].shape)
            return dict_out, target_out, id
        else:
            return img_out, target_out, id

    def __len__(self):
        return len(self.study_ids)

    ##
    ## INITIALIZATION AND TRAINING
    ##


def init_weights(m):
    #if type(m) == nn.Linear:
    print(m.weight)
    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)
    print(m.weight)


def train(args, data_dir, train_dict, test_dict, st_dict, stan_dict, ui_dict, chop_dict, prenatal_dict, max_epochs, cov_in, rand_crop=False, train_only=False):

    if train_only:
        if args.final_layers:
            out_file_root = args.out_dir + "/FinalLayerFineTune" + str(args.out_label) + "_TrainOnly_" + str(max_epochs) + "epochs_bs" + \
                            str(args.batch_size) + "_lr" + str(args.lr) + \
                            "_RC" + str(args.random_crop) + "_cov" + str(cov_in) + "_OS" + str(args.ordered_split)
            out_file_name = out_file_root + ".txt"
        else:
            out_file_root = args.out_dir + "/NoFinalLayerFineTune" + str(args.out_label) + "_TrainOnly_" + str(max_epochs) + "epochs_bs" + \
                            str(args.batch_size) + "_lr" + str(args.lr) + \
                            "_RC" + str(args.random_crop) + "_cov" + str(cov_in) + "_OS" + str(args.ordered_split)
            out_file_name = out_file_root + ".txt"

    else:
        out_file_root = args.out_dir + "/" + str(args.out_label) + "_" + str(max_epochs) + "epochs_bs" + \
                        str(args.batch_size) + "_lr" + str(args.lr) + \
                        "_RC" + str(args.random_crop) + "_cov" + str(cov_in) + "_OS" + str(args.ordered_split)
        out_file_name = out_file_root + ".txt"


    outfile = open(out_file_name, 'w+')
    outfile.close()

    hyperparams = {'lr': args.lr,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay,
                   'train/test_split': args.split 
                  }
    # params = {'batch_size': args.batch_size,
    #           'shuffle': True,
    #           'num_workers': args.num_workers}

    if not train_only:
        test_set = KidneyDataset(in_dict=test_dict, data_dir=data_dir, cov_input=cov_in, ablate_age=args.ablate_age, ablate_side=args.ablate_side)
        test_generator = DataLoader(test_set, num_workers=0, batch_size=16)

    st_test_set = KidneyDataset(in_dict=st_dict, data_dir=data_dir, cov_input=cov_in, silent_trial=True, ablate_age=args.ablate_age, ablate_side=args.ablate_side)
    st_test_generator = DataLoader(st_test_set, num_workers=0, batch_size=16)

    stan_test_set = KidneyDataset(in_dict=stan_dict, data_dir=data_dir, cov_input=cov_in, crop=True, ablate_age=args.ablate_age, ablate_side=args.ablate_side)
    stan_test_generator = DataLoader(stan_test_set, num_workers=0, batch_size=16)

    ui_test_set = KidneyDataset(in_dict=ui_dict, data_dir=data_dir, cov_input=cov_in, crop=True, ablate_age=args.ablate_age, ablate_side=args.ablate_side)
    ui_test_generator = DataLoader(ui_test_set, num_workers=0, batch_size=16)

    chop_test_set = KidneyDataset(in_dict=chop_dict, data_dir=data_dir, cov_input=cov_in, crop=True, ablate_age=args.ablate_age, ablate_side=args.ablate_side)
    chop_test_generator = DataLoader(chop_test_set, num_workers=0, batch_size=16)

    # prenatal_test_set = KidneyDataset(in_dict=prenatal_dict, data_dir=data_dir, cov_input=cov_in, crop=True)
    # prenatal_test_generator = DataLoader(prenatal_test_set, num_workers=0, batch_size=16)

    if train_only:
        ## make tuple
        train_img_dict, train_label_dict, train_cov_dict, train_study_ids = get_images(train_dict, data_dir=data_dir, random_crop=rand_crop, update_num=1, ablate_age=args.ablate_age)
        train_study_ids = list(train_study_ids)
        # train_study_ids = shuffle(list(train_study_ids), random_state=42)

        fold = 1
        n_splits = 1
        split = 0
        model_output = {"train": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "val": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "st": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "stan": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "ui": {str(j + 1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "chop": {str(j + 1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "prenatal": {str(j + 1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)}}

        skf = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        id_tuples = skf.split(X=train_study_ids, y=list(train_label_dict.values()))

    else:
        fold = 1
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # train_y = np.array(train_y)
        # train_cov = np.array(train_cov)

        if rand_crop:
            train_img_dict, train_label_dict, train_cov_dict, train_study_ids = get_images(train_dict, data_dir=data_dir,
                                                                                           random_crop=rand_crop,
                                                                                           update_num=1, ablate_age=args.ablate_age)
            train_img_dict2, train_label_dict2, train_cov_dict2, train_study_ids2 = get_images(train_dict, data_dir=data_dir,
                                                                                           random_crop=rand_crop,
                                                                                           update_num=2, update=True, ablate_age=args.ablate_age)
            train_img_dict.update(train_img_dict2)
            train_label_dict.update(train_label_dict2)
            train_cov_dict.update(train_cov_dict2)
            # train_study_ids.update(train_study_ids2)

            train_img_dict3, train_label_dict3, train_cov_dict3, train_study_ids3 = get_images(train_dict, data_dir=data_dir,
                                                                                           random_crop=rand_crop,
                                                                                           update_num=3, update=True, ablate_age=args.ablate_age)
            train_img_dict.update(train_img_dict3)
            train_label_dict.update(train_label_dict3)
            train_cov_dict.update(train_cov_dict3)
            # train_study_ids.update(train_study_ids3)

            train_study_ids = train_img_dict.keys()

        else:
            train_img_dict, train_label_dict, train_cov_dict, train_study_ids = get_images(train_dict, data_dir=data_dir, ablate_age=args.ablate_age)

        # train_study_ids = shuffle(list(train_study_ids), random_state=42)
        train_study_ids = list(train_study_ids)

        model_output = {"train": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "val": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "test": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "st": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "stan": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "ui": {str(j+1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "chop": {str(j + 1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)},
                        "prenatal": {str(j + 1): {str(k): dict() for k in range(max_epochs)} for j in range(n_splits)}}
        split = 0

        id_tuples = skf.split(train_study_ids, list(train_label_dict.values()))

    for train_tuple in id_tuples:

        train_idx, val_idx = train_tuple

        split += 1

        if args.view != "siamese":
            net = SiamNet(num_inputs=1, cov_layers=cov_in, one_hot=args.one_hot).to(device)
        else:
            if args.alt_model: ## add alt_model as flag
                net = SiamNet_v2(cov_layers=cov_in, one_hot=args.one_hot).to(device)
            else:
                net = SiamNet(cov_layers=cov_in, one_hot=args.one_hot).to(device)
        net.zero_grad()
        #net.apply(init_weights)

        if args.checkpoint != "":
            # pretrained_dict = torch.load(args.checkpoint)
            print("loading checkpoint..............")
            pretrained_dict = torch.load(args.checkpoint)['model_state_dict']
            model_dict = net.state_dict()

            for k, v in model_dict.items():
              if k in pretrained_dict:
                  # print(k)
                  model_dict[k] = pretrained_dict[k]

            # 2. overwrite entries in the existing state dict
            # model_dict.update(pretrained_dict)

            # 3. load the new state dict
            net.load_state_dict(model_dict)
            print("Checkpoint loaded...............")

            if args.final_layers:
                # for param in net.parameters():
                #     print(param)
                i = 0
                for child in net.children():
                    i += 1
                    # print(i)
                    # print(child)
                    if i < 6:
                        for param in child.parameters():
                            param.requires_grad = False

        if args.adam:
            optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                         weight_decay=hyperparams['weight_decay'])
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                        weight_decay=hyperparams['weight_decay'])

        print("Splitting Training/Validation data")

        train_ids = [train_study_ids[k] for k in train_idx]
        val_ids = [train_study_ids[k] for k in val_idx]

        if not train_only:
            _, _, _, test_ids = get_images(test_dict)
            print(test_ids)

        train_imgs_d = {train_id: train_img_dict[train_id] for train_id in train_ids}
        val_imgs_d = {val_id: train_img_dict[val_id] for val_id in val_ids}

        train_lab_d = {train_id: train_label_dict[train_id] for train_id in train_ids}
        val_lab_d = {val_id: train_label_dict[val_id] for val_id in val_ids}

        train_cov_d = {train_id: train_cov_dict[train_id] for train_id in train_ids}
        val_cov_d = {val_id: train_cov_dict[val_id] for val_id in val_ids}

        training_set = KidneyDataset(from_full_dict=False, image_dict=train_imgs_d, label_dict=train_lab_d, cov_dict=train_cov_d,
                                     study_ids=train_ids, rand_crop=rand_crop, cov_input=cov_in, ablate_age=args.ablate_age, ablate_side=args.ablate_side)
        training_generator = DataLoader(training_set, shuffle=True, num_workers=0, batch_size=16)

        ### DEBUGGING DATA LOADER FEB 22, 2022
        # for my_key in train_imgs_d.keys():
        #     if train_imgs_d[my_key]['trv'].shape != (256, 256):
        #         print(train_imgs_d[my_key]['trv'].shape)
        #
        # for my_key in train_lab_d.keys():
        #     if train_lab_d[my_key] != 1 & train_lab_d[my_key] != 0:
        #         print(train_lab_d[my_key])
        #
        # for my_key in train_cov_d.keys():
        #     print(train_cov_d[my_key]['Sex'])
        #     print(train_cov_d[my_key]['Age_wks'])
        #
        #
        # set(train_imgs_d.keys()) == set(train_lab_d.keys())
        # set(train_imgs_d.keys()) == set(train_cov_d.keys())
        #
        # set(training_set.image_dict.keys()) == set(training_set.label_dict.keys())
        # training_set.cov_dict.keys()

        print("Training set ready.")

        validation_set = KidneyDataset(from_full_dict=False, image_dict=val_imgs_d, label_dict=val_lab_d, cov_dict=val_cov_d,
                                       study_ids=val_ids, rand_crop=False, cov_input=cov_in, ablate_age=args.ablate_age, ablate_side=args.ablate_side)
        validation_generator = DataLoader(validation_set, num_workers=0, batch_size=16)

        print("Validation set ready.")

    ## run all epochs on given split of the data
        for epoch in range(max_epochs):
            print("Epoch: " + str(epoch) + " running.")
            accurate_labels_train = 0
            accurate_labels_val = 0
            accurate_labels_test = 0
            accurate_labels_st = 0
            accurate_labels_stan = 0
            accurate_labels_ui = 0
            accurate_labels_chop = 0
            accurate_labels_prenatal = 0

            loss_accum_train = 0
            loss_accum_val = 0
            loss_accum_test = 0
            loss_accum_st = 0
            loss_accum_stan = 0
            loss_accum_ui = 0
            loss_accum_chop = 0
            loss_accum_prenatal = 0

            all_targets_train = []
            all_pred_prob_train = []
            all_pred_label_train = []

            all_targets_val = []
            all_pred_prob_val = []
            all_pred_label_val = []

            all_targets_test = []
            all_pred_prob_test = []
            all_pred_label_test = []

            all_targets_st = []
            all_pred_prob_st = []
            all_pred_label_st = []

            all_targets_stan = []
            all_pred_prob_stan = []
            all_pred_label_stan = []

            all_targets_ui = []
            all_pred_prob_ui = []
            all_pred_label_ui = []

            all_targets_chop = []
            all_pred_prob_chop = []
            all_pred_label_chop = []

            all_targets_prenatal = []
            all_pred_prob_prenatal = []
            all_pred_label_prenatal = []

            patient_ID_train = []
            patient_ID_val = []
            patient_ID_test = []
            patient_ID_st = []
            patient_ID_stan = []
            patient_ID_ui = []
            patient_ID_chop = []
            patient_ID_prenatal = []

            counter_train = 0
            counter_val = 0
            counter_test = 0
            counter_st = 0
            counter_stan = 0
            counter_ui = 0
            counter_chop = 0
            counter_prenatal = 0

            net.train()

            ## Run training set
            for batch_idx, (data, target, id) in enumerate(training_generator):
                # print("Training batch drawn.")
                optimizer.zero_grad()
                #net.train() # 20190619
                # print("Running network.")
                if (epoch+1 == args.embed_epoch or epoch+1 == 1 or epoch+1 == 10) and args.embed:
                    output, train_embed = net(data, get_embeddings=args.embed)
                    embd_pd = pd.DataFrame(train_embed.cpu().detach().numpy())
                    embd_pd['id'] = id
                    embd_pd.to_csv(args.out_dir + "/Training_epoch" + str(epoch+1) + "_embeddings_" + str(args.out_label) + ".csv", mode='a', header=False)
                else:
                    output = net(data)

                # print("Target prepared.")
                target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)
                #print(output.shape, target.shape)

                # print("Output: ")
                # print(output)
                # print("Target: ")
                # print(target)

                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                loss = F.cross_entropy(output, target)
                #loss = cross_entropy(output, target)
                # print("Loss: ")
                # print(loss)
                loss_accum_train += loss.item() * len(target)

                loss.backward()

                accurate_labels_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                optimizer.step()
                counter_train += len(target)

                output_softmax = softmax(output)
                #output_softmax = output
                pred_prob = output_softmax[:, 1]
                pred_label = torch.argmax(output_softmax, dim=1)

                # print("Training")
                # print(target)
                # print(pred_prob)
                # print(len(target))
                # print(len(pred_prob))

                # print(pred_label)
                assert len(pred_prob) == len(target)
                assert len(pred_label) == len(target)


                all_pred_prob_train.append(pred_prob)
                all_targets_train.append(target)
                all_pred_label_train.append(pred_label)
                patient_ID_train.append(list(id))

            net.eval()

            ## Run val set
            with torch.no_grad():
            #with torch.set_grad_enabled(False):
                for batch_idx, (data, target, id) in enumerate(validation_generator):
                    # print("Validation batch drawn.")
                    net.zero_grad()
                    #net.eval() # 20190619
                    optimizer.zero_grad()

                    if (epoch+1 == args.embed_epoch or epoch+1 == 1 or epoch+1 == 10) and args.embed:
                        output, val_embed = net(data, get_embeddings=args.embed)
                        embd_pd = pd.DataFrame(val_embed.cpu().detach().numpy())
                        embd_pd['id'] = id
                        embd_pd.to_csv(args.out_dir + "/Validation_epoch" + str(epoch+1) + "_embeddings_" + str(args.out_label) + ".csv", mode='a', header=False)
                    else:
                        output = net(data)

                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    #loss = cross_entropy(output, target)
                    loss_accum_val += loss.item() * len(target)
                    counter_val += len(target)
                    #output_softmax = output
                    output_softmax = softmax(output)

                    accurate_labels_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:, 1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    # print("Validating")
                    # print(target)
                    # print(pred_prob)
                    # print(len(target))
                    # print(len(pred_prob))

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_val.append(pred_prob)
                    all_targets_val.append(target)
                    all_pred_label_val.append(pred_label)
                    patient_ID_val.append(list(id))
            net.eval()

            ## Run test set
            if not train_only:
                with torch.no_grad():
                #with torch.set_grad_enabled(False):
                    for batch_idx, (data, target, id) in enumerate(test_generator):
                        # print("Test batch drawn.")
                        net.zero_grad()
                        net.eval() # 20190619
                        optimizer.zero_grad()

                        if (epoch+1 == args.embed_epoch or epoch+1 == 1 or epoch+1 == 10) and args.embed:
                            output, test_embed = net(data, get_embeddings=args.embed)
                            embd_pd = pd.DataFrame(test_embed.cpu().detach().numpy())
                            embd_pd['id'] = id
                            embd_pd.to_csv(args.out_dir + "/Test_epoch" + str(epoch+1) + "_embeddings_" + str(args.out_label) + ".csv", mode='a', header=False)
                        else:
                            output = net(data)

                        target = target.type(torch.LongTensor).to(device)

                        loss = F.cross_entropy(output, target)
                        #loss = cross_entropy(output, target)
                        loss_accum_test += loss.item() * len(target)
                        counter_test += len(target)
                        output_softmax = softmax(output)
                        #output_softmax = output
                        accurate_labels_test += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                        pred_prob = output_softmax[:, 1]
                        pred_prob = pred_prob.squeeze()
                        pred_label = torch.argmax(output, dim=1)

                        # print("Testing")
                        # print(target)
                        # print(pred_prob)
                        # print(len(target))
                        # print(len(pred_prob))

                        if pred_prob.shape == torch.Size([]):
                            pred_prob = pred_prob.unsqueeze(0)

                        assert len(pred_prob) == len(target)
                        assert len(pred_label) == len(target)

                        all_pred_prob_test.append(pred_prob)
                        all_targets_test.append(target)
                        all_pred_label_test.append(pred_label)

                        patient_ID_test.append(list(id))

            ## Run silent trial test set
            with torch.no_grad():
            #with torch.set_grad_enabled(False):
                for batch_idx, (data, target, id) in enumerate(st_test_generator):
                    # print("Test batch drawn.")
                    net.zero_grad()
                    net.eval() # 20190619
                    optimizer.zero_grad()

                    if (epoch+1 == args.embed_epoch or epoch+1 == 1 or epoch+1 == 10) and args.embed:
                        output, test_embed = net(data, get_embeddings=args.embed)
                        embd_pd = pd.DataFrame(test_embed.cpu().detach().numpy())
                        embd_pd['id'] = id
                        embd_pd.to_csv(args.out_dir + "/SilentTrial_epoch" + str(epoch+1) + "_embeddings_" + str(args.out_label) + ".csv", mode='a', header=False)
                    else:
                        output = net(data)

                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    #loss = cross_entropy(output, target)
                    loss_accum_st += loss.item() * len(target)
                    counter_st += len(target)
                    output_softmax = softmax(output)
                    #output_softmax = output
                    accurate_labels_st += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:, 1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    # print("Testing")
                    # print(target)
                    # print(pred_prob)
                    # print(len(target))
                    # print(len(pred_prob))

                    if pred_prob.shape == torch.Size([]):
                        pred_prob = pred_prob.unsqueeze(0)

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_st.append(pred_prob)
                    all_targets_st.append(target)
                    all_pred_label_st.append(pred_label)

                    patient_ID_st.append(list(id))

            ## Run stanford test set
            with torch.no_grad():
            #with torch.set_grad_enabled(False):
                for batch_idx, (data, target, id) in enumerate(stan_test_generator):
                    # print("Test batch drawn.")
                    net.zero_grad()
                    net.eval() # 20190619
                    optimizer.zero_grad()

                    if (epoch+1 == args.embed_epoch or epoch+1 == 1 or epoch+1 == 10) and args.embed:
                        output, test_embed = net(data, get_embeddings=args.embed)
                        embd_pd = pd.DataFrame(test_embed.cpu().detach().numpy())
                        embd_pd['id'] = id
                        embd_pd.to_csv(args.out_dir + "/Stanford_epoch" + str(epoch+1) + "_embeddings_" + str(args.out_label) + ".csv", mode='a', header=False)
                    else:
                        output = net(data)

                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    #loss = cross_entropy(output, target)
                    loss_accum_stan += loss.item() * len(target)
                    counter_stan += len(target)
                    output_softmax = softmax(output)
                    #output_softmax = output
                    accurate_labels_stan += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:, 1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    # print("Testing")
                    # print(target)
                    # print(pred_prob)
                    # print(len(target))
                    # print(len(pred_prob))

                    if pred_prob.shape == torch.Size([]):
                        pred_prob = pred_prob.unsqueeze(0)

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_stan.append(pred_prob)
                    all_targets_stan.append(target)
                    all_pred_label_stan.append(pred_label)

                    patient_ID_stan.append(list(id))

            ## Run UIowa test set
            with torch.no_grad():
            #with torch.set_grad_enabled(False):
                for batch_idx, (data, target, id) in enumerate(ui_test_generator):
                    # print("Test batch drawn.")
                    net.zero_grad()
                    net.eval() # 20190619
                    optimizer.zero_grad()

                    if (epoch+1 == args.embed_epoch or epoch+1 == 1 or epoch+1 == 10) and args.embed:
                        output, test_embed = net(data, get_embeddings=args.embed)
                        embd_pd = pd.DataFrame(test_embed.cpu().detach().numpy())
                        embd_pd['id'] = id
                        embd_pd.to_csv(args.out_dir + "/UIowa_epoch" + str(epoch+1) + "_embeddings_" + str(args.out_label) + ".csv", mode='a', header=False)
                    else:
                        output = net(data)

                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    #loss = cross_entropy(output, target)
                    loss_accum_ui += loss.item() * len(target)
                    counter_ui += len(target)
                    output_softmax = softmax(output)
                    #output_softmax = output
                    accurate_labels_ui += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:, 1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    # print("Testing")
                    # print(target)
                    # print(pred_prob)
                    # print(len(target))
                    # print(len(pred_prob))

                    if pred_prob.shape == torch.Size([]):
                        pred_prob = pred_prob.unsqueeze(0)

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_ui.append(pred_prob)
                    all_targets_ui.append(target)
                    all_pred_label_ui.append(pred_label)

                    patient_ID_ui.append(list(id))

            ## Run CHOP test set
            with torch.no_grad():
            #with torch.set_grad_enabled(False):
                for batch_idx, (data, target, id) in enumerate(chop_test_generator):
                    # print("Test batch drawn.")
                    net.zero_grad()
                    net.eval() # 20190619
                    optimizer.zero_grad()

                    if (epoch+1 == args.embed_epoch or epoch+1 == 1 or epoch+1 == 10) and args.embed:
                        output, test_embed = net(data, get_embeddings=args.embed)
                        embd_pd = pd.DataFrame(test_embed.cpu().detach().numpy())
                        embd_pd['id'] = id
                        embd_pd.to_csv(args.out_dir + "/CHOP_epoch" + str(epoch+1) + "_embeddings_" + str(args.out_label) + ".csv", mode='a', header=False)
                    else:
                        output = net(data)

                    target = target.type(torch.LongTensor).to(device)

                    loss = F.cross_entropy(output, target)
                    #loss = cross_entropy(output, target)
                    loss_accum_chop += loss.item() * len(target)
                    counter_chop += len(target)
                    output_softmax = softmax(output)
                    #output_softmax = output
                    accurate_labels_chop += torch.sum(torch.argmax(output, dim=1) == target).cpu()

                    pred_prob = output_softmax[:, 1]
                    pred_prob = pred_prob.squeeze()
                    pred_label = torch.argmax(output, dim=1)

                    # print("Testing")
                    # print(target)
                    # print(pred_prob)
                    # print(len(target))
                    # print(len(pred_prob))

                    if pred_prob.shape == torch.Size([]):
                        pred_prob = pred_prob.unsqueeze(0)

                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)

                    all_pred_prob_chop.append(pred_prob)
                    all_targets_chop.append(target)
                    all_pred_label_chop.append(pred_label)

                    patient_ID_chop.append(list(id))

            ## Run prenatal test set
            # with torch.no_grad():
            # #with torch.set_grad_enabled(False):
            #     for batch_idx, (data, target, id) in enumerate(prenatal_test_generator):
            #         # print("Test batch drawn.")
            #         net.zero_grad()
            #         net.eval() # 20190619
            #         optimizer.zero_grad()
            #
            #         if epoch == args.embed_epoch and args.embed:
            #             output, test_embed = net(data, get_embeddings=args.embed)
            #             embd_pd = pd.DataFrame(test_embed.cpu().detach().numpy())
            #             embd_pd['id'] = id
            #             embd_pd.to_csv(args.out_dir + "/Prenatal_epoch" + str(args.embed_epoch) + "_embeddings.csv", mode='a', header=False)
            #         else:
            #             output = net(data)
            #
            #         target = target.type(torch.LongTensor).to(device)
            #
            #         loss = F.cross_entropy(output, target)
            #         #loss = cross_entropy(output, target)
            #         loss_accum_prenatal += loss.item() * len(target)
            #         counter_prenatal += len(target)
            #         output_softmax = softmax(output)
            #         #output_softmax = output
            #         accurate_labels_chop += torch.sum(torch.argmax(output, dim=1) == target).cpu()
            #
            #         pred_prob = output_softmax[:, 1]
            #         pred_prob = pred_prob.squeeze()
            #         pred_label = torch.argmax(output, dim=1)
            #
            #         # print("Testing")
            #         # print(target)
            #         # print(pred_prob)
            #         # print(len(target))
            #         # print(len(pred_prob))
            #
            #         if pred_prob.shape == torch.Size([]):
            #             pred_prob = pred_prob.unsqueeze(0)
            #
            #         assert len(pred_prob) == len(target)
            #         assert len(pred_label) == len(target)
            #
            #         all_pred_prob_prenatal.append(pred_prob)
            #         all_targets_prenatal.append(target)
            #         all_pred_label_prenatal.append(pred_label)
            #
            #         patient_ID_prenatal.append(list(id))

            all_pred_prob_train = torch.cat(all_pred_prob_train)
            all_targets_train = torch.cat(all_targets_train)
            all_pred_label_train = torch.cat(all_pred_label_train)
            all_train_ids = flatten(patient_ID_train)

            model_output['train'][str(split)][str(epoch)]['id'] = all_train_ids
            model_output['train'][str(split)][str(epoch)]['pred'] = all_pred_prob_train.tolist()
            model_output['train'][str(split)][str(epoch)]['target'] = all_targets_train.tolist()

           # patient_ID_train = torch.cat(patient_ID_train)

            assert len(all_targets_train) == len(training_set)
            assert len(all_pred_prob_train) == len(training_set)
            assert len(all_pred_label_train) == len(training_set)
            # assert len(patient_ID_train) == len(training_set)

            results_train = process_results.get_metrics(y_score=all_pred_prob_train.cpu().detach().numpy(),
                                                  y_true=all_targets_train.cpu().detach().numpy(),
                                                  y_pred=all_pred_label_train.cpu().detach().numpy())

            print('Fold\t{}\tTrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_train)/counter_train,
                                                                        loss_accum_train/counter_train, results_train['auc'],
                                                                        results_train['auprc'], results_train['tn'],
                                                                        results_train['fp'], results_train['fn'],
                                                                        results_train['tp']))
            outfile = open(out_file_name, 'a')
            outfile.write('\nFold\t{}\tTrainEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_train)/counter_train,
                                                                        loss_accum_train/counter_train, results_train['auc'],
                                                                        results_train['auprc'], results_train['tn'],
                                                                        results_train['fp'], results_train['fn'],
                                                                        results_train['tp']))
            outfile.close()

            all_pred_prob_val = torch.cat(all_pred_prob_val)
            all_targets_val = torch.cat(all_targets_val)
            all_pred_label_val = torch.cat(all_pred_label_val)
            all_val_ids = flatten(patient_ID_val)

            model_output['val'][str(split)][str(epoch)]['id'] = all_val_ids
            model_output['val'][str(split)][str(epoch)]['pred'] = all_pred_prob_val.tolist()
            model_output['val'][str(split)][str(epoch)]['target'] = all_targets_val.tolist()

            #patient_ID_val = torch.cat(patient_ID_val)

            assert len(all_targets_val) == len(validation_set)
            assert len(all_pred_prob_val) == len(validation_set)
            assert len(all_pred_label_val) == len(validation_set)
            # assert len(patient_ID_val) == len(validation_set)

            results_val = process_results.get_metrics(y_score=all_pred_prob_val.cpu().detach().numpy(),
                                                      y_true=all_targets_val.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_val.cpu().detach().numpy())
            print('Fold\t{}\tValEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_val) / counter_val,
                                                                         loss_accum_val / counter_val, results_val['auc'],
                                                                         results_val['auprc'], results_val['tn'],
                                                                         results_val['fp'], results_val['fn'],
                                                                         results_val['tp']))
            outfile = open(out_file_name, 'a')
            outfile.write('\nFold\t{}\tValEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_val) / counter_val,
                                                                         loss_accum_val / counter_val, results_val['auc'],
                                                                         results_val['auprc'], results_val['tn'],
                                                                         results_val['fp'], results_val['fn'],
                                                                         results_val['tp']))
            outfile.close()

            if not train_only:
                all_pred_prob_test = torch.cat(all_pred_prob_test)
                all_targets_test = torch.cat(all_targets_test)
                all_pred_label_test = torch.cat(all_pred_label_test)
                all_test_ids = flatten(patient_ID_test)

                model_output['test'][str(split)][str(epoch)]['id'] = all_test_ids
                model_output['test'][str(split)][str(epoch)]['pred'] = all_pred_prob_test.tolist()
                model_output['test'][str(split)][str(epoch)]['target'] = all_targets_test.tolist()

                # patient_ID_test = torch.cat(patient_ID_test)

                assert len(all_targets_test) == len(test_set)
                assert len(all_pred_label_test) == len(test_set)
                assert len(all_pred_prob_test) == len(test_set)
                # assert len(patient_ID_test) == len(test_set)

                results_test = process_results.get_metrics(y_score=all_pred_prob_test.cpu().detach().numpy(),
                                                          y_true=all_targets_test.cpu().detach().numpy(),
                                                          y_pred=all_pred_label_test.cpu().detach().numpy())
                print('Fold\t{}\tTestEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                      'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_test) / counter_test,
                                                                             loss_accum_test / counter_test, results_test['auc'],
                                                                             results_test['auprc'], results_test['tn'],
                                                                             results_test['fp'], results_test['fn'],
                                                                             results_test['tp']))
                outfile = open(out_file_name, 'a')
                outfile.write('\nFold\t{}\tTestEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                              'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_test) / counter_test,
                                                                             loss_accum_test / counter_test, results_test['auc'],
                                                                             results_test['auprc'], results_test['tn'],
                                                                             results_test['fp'], results_test['fn'],
                                                                             results_test['tp']))
                outfile.close()

            all_pred_prob_st = torch.cat(all_pred_prob_st)
            all_targets_st = torch.cat(all_targets_st)
            all_pred_label_st = torch.cat(all_pred_label_st)
            all_st_ids = flatten(patient_ID_st)

            model_output['st'][str(split)][str(epoch)]['id'] = all_st_ids
            model_output['st'][str(split)][str(epoch)]['pred'] = all_pred_prob_st.tolist()
            model_output['st'][str(split)][str(epoch)]['target'] = all_targets_st.tolist()

            # patient_ID_test = torch.cat(patient_ID_test)

            # assert len(all_targets_st) == len(test_set)
            # assert len(all_pred_label_st) == len(test_set)
            # assert len(all_pred_prob_st) == len(test_set)
            # assert len(patient_ID_test) == len(test_set)

            results_st = process_results.get_metrics(y_score=all_pred_prob_st.cpu().detach().numpy(),
                                                      y_true=all_targets_st.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_st.cpu().detach().numpy())
            print('Fold\t{}\tSilentTrialEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_st) / counter_st,
                                                                         loss_accum_st / counter_st, results_st['auc'],
                                                                         results_st['auprc'], results_st['tn'],
                                                                         results_st['fp'], results_st['fn'],
                                                                         results_st['tp']))
            outfile = open(out_file_name, 'a')
            outfile.write('Fold\t{}\tSilentTrialEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_st) / counter_st,
                                                                         loss_accum_st / counter_st, results_st['auc'],
                                                                         results_st['auprc'], results_st['tn'],
                                                                         results_st['fp'], results_st['fn'],
                                                                         results_st['tp']))
            outfile.close()

            all_pred_prob_stan = torch.cat(all_pred_prob_stan)
            all_targets_stan = torch.cat(all_targets_stan)
            all_pred_label_stan = torch.cat(all_pred_label_stan)
            all_stan_ids = flatten(patient_ID_stan)

            model_output['stan'][str(split)][str(epoch)]['id'] = all_stan_ids
            model_output['stan'][str(split)][str(epoch)]['pred'] = all_pred_prob_stan.tolist()
            model_output['stan'][str(split)][str(epoch)]['target'] = all_targets_stan.tolist()

            # patient_ID_test = torch.cat(patient_ID_test)

            # assert len(all_targets_st) == len(test_set)
            # assert len(all_pred_label_st) == len(test_set)
            # assert len(all_pred_prob_st) == len(test_set)
            # assert len(patient_ID_test) == len(test_set)

            results_stan = process_results.get_metrics(y_score=all_pred_prob_stan.cpu().detach().numpy(),
                                                      y_true=all_targets_stan.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_stan.cpu().detach().numpy())
            print('Fold\t{}\tStanfordEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_stan) / counter_stan,
                                                                         loss_accum_stan / counter_stan, results_stan['auc'],
                                                                         results_stan['auprc'], results_stan['tn'],
                                                                         results_stan['fp'], results_stan['fn'],
                                                                         results_stan['tp']))
            outfile = open(out_file_name, 'a')
            outfile.write('Fold\t{}\tStanfordEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_stan) / counter_stan,
                                                                         loss_accum_stan / counter_stan, results_stan['auc'],
                                                                         results_stan['auprc'], results_stan['tn'],
                                                                         results_stan['fp'], results_stan['fn'],
                                                                         results_stan['tp']))
            outfile.close()


            all_pred_prob_ui = torch.cat(all_pred_prob_ui)
            all_targets_ui = torch.cat(all_targets_ui)
            all_pred_label_ui = torch.cat(all_pred_label_ui)
            all_ui_ids = flatten(patient_ID_ui)

            model_output['ui'][str(split)][str(epoch)]['id'] = all_ui_ids
            model_output['ui'][str(split)][str(epoch)]['pred'] = all_pred_prob_ui.tolist()
            model_output['ui'][str(split)][str(epoch)]['target'] = all_targets_ui.tolist()

            # patient_ID_test = torch.cat(patient_ID_test)

            # assert len(all_targets_st) == len(test_set)
            # assert len(all_pred_label_st) == len(test_set)
            # assert len(all_pred_prob_st) == len(test_set)
            # assert len(patient_ID_test) == len(test_set)

            results_ui = process_results.get_metrics(y_score=all_pred_prob_ui.cpu().detach().numpy(),
                                                      y_true=all_targets_ui.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_ui.cpu().detach().numpy())
            print('Fold\t{}\tUIowaEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_ui) / counter_ui,
                                                                         loss_accum_ui / counter_ui, results_ui['auc'],
                                                                         results_ui['auprc'], results_ui['tn'],
                                                                         results_ui['fp'], results_ui['fn'],
                                                                         results_ui['tp']))
            outfile = open(out_file_name, 'a')
            outfile.write('Fold\t{}\tUIowaEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_ui) / counter_ui,
                                                                         loss_accum_ui / counter_ui, results_ui['auc'],
                                                                         results_ui['auprc'], results_ui['tn'],
                                                                         results_ui['fp'], results_ui['fn'],
                                                                         results_ui['tp']))
            outfile.close()


            all_pred_prob_chop = torch.cat(all_pred_prob_chop)
            all_targets_chop = torch.cat(all_targets_chop)
            all_pred_label_chop = torch.cat(all_pred_label_chop)
            all_chop_ids = flatten(patient_ID_chop)

            model_output['chop'][str(split)][str(epoch)]['id'] = all_chop_ids
            model_output['chop'][str(split)][str(epoch)]['pred'] = all_pred_prob_chop.tolist()
            model_output['chop'][str(split)][str(epoch)]['target'] = all_targets_chop.tolist()

            # patient_ID_test = torch.cat(patient_ID_test)

            # assert len(all_targets_st) == len(test_set)
            # assert len(all_pred_label_st) == len(test_set)
            # assert len(all_pred_prob_st) == len(test_set)
            # assert len(patient_ID_test) == len(test_set)

            results_chop = process_results.get_metrics(y_score=all_pred_prob_chop.cpu().detach().numpy(),
                                                      y_true=all_targets_chop.cpu().detach().numpy(),
                                                      y_pred=all_pred_label_chop.cpu().detach().numpy())
            print('Fold\t{}\tCHOPEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_chop) / counter_chop,
                                                                         loss_accum_chop / counter_chop, results_chop['auc'],
                                                                         results_chop['auprc'], results_chop['tn'],
                                                                         results_chop['fp'], results_chop['fn'],
                                                                         results_chop['tp']))
            outfile = open(out_file_name, 'a')
            outfile.write('Fold\t{}\tCHOPEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
                  'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_chop) / counter_chop,
                                                                         loss_accum_chop / counter_chop, results_chop['auc'],
                                                                         results_chop['auprc'], results_chop['tn'],
                                                                         results_chop['fp'], results_chop['fn'],
                                                                         results_chop['tp']))
            outfile.close()

            # all_pred_prob_prenatal = torch.cat(all_pred_prob_prenatal)
            # all_targets_prenatal = torch.cat(all_targets_prenatal)
            # all_pred_label_prenatal = torch.cat(all_pred_label_prenatal)
            # all_prenatal_ids = flatten(patient_ID_prenatal)
            #
            # model_output['prenatal'][str(split)][str(epoch)]['id'] = all_prenatal_ids
            # model_output['prenatal'][str(split)][str(epoch)]['pred'] = all_pred_prob_prenatal.tolist()
            # model_output['prenatal'][str(split)][str(epoch)]['target'] = all_targets_prenatal.tolist()
            #
            # # patient_ID_test = torch.cat(patient_ID_test)
            #
            # # assert len(all_targets_st) == len(test_set)
            # # assert len(all_pred_label_st) == len(test_set)
            # # assert len(all_pred_prob_st) == len(test_set)
            # # assert len(patient_ID_test) == len(test_set)
            #
            # results_prenatal = process_results.get_metrics(y_score=all_pred_prob_prenatal.cpu().detach().numpy(),
            #                                           y_true=all_targets_prenatal.cpu().detach().numpy(),
            #                                           y_pred=all_pred_label_prenatal.cpu().detach().numpy())
            # print('Fold\t{}\tPrenatalEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
            #       'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_prenatal) / counter_prenatal,
            #                                                              loss_accum_prenatal / counter_prenatal, results_prenatal['auc'],
            #                                                              results_prenatal['auprc'], results_prenatal['tn'],
            #                                                              results_prenatal['fp'], results_prenatal['fn'],
            #                                                              results_prenatal['tp']))
            # outfile = open(out_file_name, 'a')
            # outfile.write('Fold\t{}\tPrenatalEpoch\t{}\tACC\t{:.6f}\tLoss\t{:.6f}\tAUC\t{:.6f}\t'
            #       'AUPRC\t{:.6f}\tTN\t{}\tFP\t{}\tFN\t{}\tTP\t{}'.format(fold, epoch, int(accurate_labels_prenatal) / counter_prenatal,
            #                                                              loss_accum_prenatal / counter_prenatal, results_prenatal['auc'],
            #                                                              results_prenatal['auprc'], results_prenatal['tn'],
            #                                                              results_prenatal['fp'], results_prenatal['fn'],
            #                                                              results_prenatal['tp']))
            # outfile.close()

            # if ((epoch+1) % 5) == 0 and epoch > 0:
            if train_only:
                checkpoint = {'epoch': epoch,
                              'loss': loss,
                              'hyperparams': hyperparams,
                              'args': args,
                              'model_state_dict': net.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'loss_train': loss_accum_train / counter_train,
                              'loss_val': loss_accum_val / counter_val,
                              'accuracy_train': int(accurate_labels_train) / counter_train,
                              'accuracy_val': int(accurate_labels_val) / counter_val,
                              'results_train': results_train,
                              'results_val': results_val,
                              'all_pred_prob_train': all_pred_prob_train,
                              'all_pred_prob_val': all_pred_prob_val,
                              'all_targets_train': all_targets_train,
                              'all_targets_val': all_targets_val,
                              'patient_ID_train': patient_ID_train,
                              'patient_ID_val': patient_ID_val,
                              'patient_ID_test': patient_ID_test}

            else:
                checkpoint = {'epoch': epoch,
                              'loss': loss,
                              'hyperparams': hyperparams,
                              'args': args,
                              'model_state_dict': net.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'loss_train': loss_accum_train / counter_train,
                              'loss_val': loss_accum_val / counter_val,
                              'loss_test': loss_accum_test / counter_test,
                              'accuracy_train': int(accurate_labels_train) / counter_train,
                              'accuracy_val': int(accurate_labels_val) / counter_val,
                              'accuracy_test': int(accurate_labels_test) / counter_test,
                              'results_train': results_train,
                              'results_val': results_val,
                              'results_test': results_test,
                              'all_pred_prob_train': all_pred_prob_train,
                              'all_pred_prob_val': all_pred_prob_val,
                              'all_pred_prob_test': all_pred_prob_test,
                              'all_targets_train': all_targets_train,
                              'all_targets_val': all_targets_val,
                              'all_targets_test': all_targets_test,
                              'patient_ID_train': patient_ID_train,
                              'patient_ID_val': patient_ID_val,
                              'patient_ID_test': patient_ID_test}

            if not os.path.isdir(args.dir):
                os.makedirs(args.dir)
            #if not os.path.isdir(args.dir + "/" + str(fold)):
                #os.makedirs(args.dir + "/" + str(fold))

            # if epoch == 9:
            #     path_to_checkpoint = out_file_root + '_10thEpoch.pth'
            #     torch.save(checkpoint, path_to_checkpoint)
            # if epoch == 12:
            #     path_to_checkpoint = out_file_root + '_13thEpoch.pth'
            #     torch.save(checkpoint, path_to_checkpoint)
            # if epoch == 15:
            #     path_to_checkpoint = out_file_root + '_16thEpoch.pth'
            #     torch.save(checkpoint, path_to_checkpoint)

            ## TO SAVE THE CHECKPOINT
            if epoch+1 == args.epoch_save:
                path_to_checkpoint = out_file_root + '_30thEpoch.pth'
                torch.save(checkpoint, path_to_checkpoint)
            
        fold += 1

    with open(out_file_root + ".json", "w") as fp:
        json.dump(model_output, fp, indent=4)

    ##
    ## MAIN FUNCTION
    ##


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int, help="Number of epochs")
    parser.add_argument('--epoch_save', default=30, type=int, help="Number of epochs at which to save the model")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    # parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--view", default="siamese", help="siamese, sag, trans")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--split", default=0.9, type=float, help="proportion of dataset to use as training")
    parser.add_argument('--ordered_split', action="store_true", default=False, help="Use Adam optimizer instead of SGD")
    parser.add_argument('--random_crop', action="store_true", default=False, help="Use Adam optimizer instead of SGD")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    parser.add_argument('--unet', action="store_true", help="UNet architecthure")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--sc", default=5, type=int, help="number of skip connections for unet (0, 1, 2, 3, 4, or 5)")
    parser.add_argument("--init", default="none")
    parser.add_argument("--hydro_only", action="store_true")
    parser.add_argument("--output_dim", default=256, type=int, help="output dim for last linear layer")
    parser.add_argument("--gender", default=None, type=str, help="choose from 'male' and 'female'")

    # parser.add_argument("--json_infile", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SKST_60%train_20210722.json",
    #                     help="Json file of model training data")
    ## original training data
    parser.add_argument("--json_infile", default="HNTrain_rootfilenames_noContra_updated20220123.json",
                        help="Json file of model training data")

    parser.add_argument("--json_st_test", default="newSTonly_rootfilenames_20220117.json",
                        help="Json file of held-out, prospective silent trial data")

    parser.add_argument("--json_stan_test", default="StanfordOnly_rootfilenames_20211229.json",
                        help="Json file of held-out, retrospective Stanford data")

    # parser.add_argument("--json_stan_test", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_Stan_finetune60%_test_20210711.json",
    #                     help="Json file of held-out, retrospective Stanford data")

    parser.add_argument("--json_ui_test", default="UIonly_moreDat_rootfilenames_20220110.json",
                        help="Json file of held-out, retrospective Stanford data")

    parser.add_argument("--json_chop_test", default="CHOP_rootfilenames_20220108.json",
                        help="Json file of held-out, retrospective CHOP data")

    parser.add_argument("--json_prenatal_test", default="Prenatal_rootfilenames_age0_20220109.json",
                        help="Json file of held-out, retrospective CHOP data")

    # parser.add_argument("--json_ui_test", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIowa_finetune60%_test_20210711.json",
    #                     help="Json file of held-out, retrospective Stanford data")

    parser.add_argument('--embed', action="store_true", default=False, help="extract embeddings")
    parser.add_argument('--embed_epoch', default=30, type=int, help="epoch to produce embeddings from")
    parser.add_argument('--cov_in', action="store_true", default=False, help="Use age at ultrasound and kidney side (R/L) as covariates in the model")
    parser.add_argument("--out_dir", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/orig_st_results/", help="Directory to save model checkpoints to")
    parser.add_argument("--train_only", action="store_true", help="Only fit train/val")
    parser.add_argument("--rand_crop_contrast", action="store_true", help="Randomly crop/contrast training dat")
    parser.add_argument("--one_hot", action="store_true", help="add one-hot encodings by site when doing a combined train")

    parser.add_argument("--data_dir", default="C:/Users/lauren erdman/OneDrive - SickKids/HN_Stanley/",
                        help="Directory where the images and data files are located")
    parser.add_argument("--final_layers", action="store_true", help="Fine-tune final layers")
        ## label for fine-tuning
    parser.add_argument("--out_label", default="NoFineTune", type=str, help="How to name model")

    parser.add_argument("--ablate_age", action="store_true", help="Ablate the 'age' covariate")
    parser.add_argument("--ablate_side", action="store_true", help="Ablate the 'side' covariate")
    parser.add_argument("--alt_model", action="store_true", help="Use alternative model w/covs")

    args = parser.parse_args()

    print("ARGS" + '\t' + str(args))

    print("break")

    if args.train_only:
        train_dict = load_dataset(args.json_infile, test_prop=0.2, ordered_split=args.ordered_split,
                                             train_only=args.train_only, data_dir=args.data_dir)
    else:
        train_dict, test_dict = load_dataset(args.json_infile, test_prop=0.2, ordered_split=args.ordered_split, train_only=args.train_only,
                                             data_dir=args.data_dir)
    # train_dict, test_dict = load_dataset(args.json_infile, test_prop=0.2, ordered_split=args.ordered_split)
    print("Data loaded.")

    st_test_dict = load_test_dataset(args.json_st_test, data_dir=args.data_dir)
    stan_test_dict = load_test_dataset(args.json_stan_test, data_dir=args.data_dir)
    ui_test_dict = load_test_dataset(args.json_ui_test, data_dir=args.data_dir)
    chop_test_dict = load_test_dataset(args.json_chop_test, data_dir=args.data_dir)
    prenatal_test_dict = load_test_dataset(args.json_prenatal_test, data_dir=args.data_dir)

    # print("break")
    # any(item in list(train_dict.keys()) for item in list(st_test_dict.keys()))

    if args.train_only:
        ## data_dir is getting overwritten between reading in the silent trial data and reading in Stanford data
                ## causing an error
        train(args, data_dir=args.data_dir, train_dict=train_dict, test_dict=None, st_dict=st_test_dict, stan_dict=stan_test_dict, ui_dict=ui_test_dict, chop_dict=chop_test_dict, prenatal_dict=prenatal_test_dict, max_epochs=args.epochs, rand_crop=args.random_crop, cov_in=args.cov_in, train_only=args.train_only)
    else:
        train(args, data_dir=args.data_dir, train_dict=train_dict, test_dict=test_dict, st_dict=st_test_dict, stan_dict=stan_test_dict, ui_dict=ui_test_dict, chop_dict=chop_test_dict, prenatal_dict=prenatal_test_dict, max_epochs=args.epochs, rand_crop=args.random_crop, cov_in=args.cov_in, train_only=args.train_only)

    # if args.view == "sag" or args.view == "trans":
    #     train_X_single=[]
    #     test_X_single=[]
    #
    #     for item in train_X:
    #         if args.view == "sag":
    #             train_X_single.append(item[0])
    #         elif args.view == "trans":
    #             train_X_single.append(item[1])
    #     for item in test_X:
    #         if args.view == "sag":
    #             test_X_single.append(item[0])
    #         elif args.view == "trans":
    #             test_X_single.append(item[1])
    #
    #     train_X=train_X_single
    #     test_X=test_X_single
    #     train_X=np.array(train_X_single)
    #     test_X=np.array(test_X_single)
    #
    #
    # print(len(train_X), len(train_y), len(train_cov), len(test_X), len(test_y), len(test_cov))
    # train(args, train_X, train_y, train_cov, test_X, test_y, test_cov, max_epochs)


if __name__ == '__main__':
    main()
