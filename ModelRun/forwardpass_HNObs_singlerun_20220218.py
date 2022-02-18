from PIL import Image
import numpy as np
import torch
import os
import argparse
from torch import nn
from skimage import exposure
from skimage import img_as_float, transform


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


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


def special_ST_preprocessing(img_file, output_dim=256):

    image = np.array(Image.open(img_file).convert('L'))
    image_grey = img_as_float(image)

    # if crop:
    #     cropped_img = crop_image(image_grey)
    # else:
    #     cropped_img = image_grey
    cropped_img = image_grey

    resized_img = transform.resize(cropped_img, output_shape=(output_dim, output_dim))
    my_img = set_contrast(resized_img) ## ultimately add contrast variable

    # img_name = img_file.split('/')[len(img_file.split('/')) - 1]
    # img_folder = "/".join(img_file.split('/')[:-2])
    # out_img_filename = img_folder + "/" + img_name.split('.')[0] + "-preprocessed.png"
    # # out_img_filename = 'C:/Users/lauren erdman/Desktop/kidney_img/img_debugging/' + img_name.split('.')[0] + "-preprocessed.png"
    # Image.fromarray(my_img * 255).convert('RGB').save(out_img_filename)
    # my_img = np.array(Image.open(out_img_filename).convert('L'))

    return my_img


def load_image(sag_path, trans_path, output_dim=256):
    X = []
    for image_path in [sag_path, trans_path]:
        print(image_path)
        img = special_ST_preprocessing(image_path, output_dim=output_dim)
        X.append(img)
    return np.array(X)


class SiamNet(nn.Module):
    def __init__(self, cov_layers, classes=2, num_inputs=2, dropout_rate=0.5, output_dim=256):
        super(SiamNet, self).__init__()

        self.cov_layers = cov_layers
        self.output_dim = output_dim
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
        self.fc7_new.add_module('fc7', nn.Linear(self.num_inputs * 512, self.output_dim))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc7_new.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc8', nn.Linear(self.output_dim, classes))

        if self.cov_layers:
            self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

            self.add_covs1 = nn.Sequential()
            self.add_covs1.add_module('fc9', nn.Linear(classes + 2, classes + 126))
            self.add_covs1.add_module('relu9', nn.ReLU(inplace=True))

            self.add_covs2 = nn.Sequential()
            self.add_covs2.add_module('fc10', nn.Linear(classes + 126, classes))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x, age=None, left=None):

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
        # x = torch.sum(x, 1)
        x = self.fc7_new(x.view(B, -1))
        pred = self.classifier_new(x)

        if self.cov_layers:
            age = torch.tensor(age).type(torch.FloatTensor).to(device).view(B, 1)
            # print("Age: ")
            # print(age)
            side = torch.tensor(left).type(torch.FloatTensor).to(device).view(B, 1)
            # print("Side: ")
            # print(side)
            mid_in = torch.cat((pred, age, side), 1)

            x = self.add_covs1(mid_in)
            pred = self.add_covs2(x)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sag_path', default='/ModelTest/Images/test_RS_cropped2.png', type=str)
    # parser.add_argument('--trans_path', default='/ModelTest/Images/test_RT_cropped2.png', type=str)
    # parser.add_argument('--age_wks', default=143, type=int)
    parser.add_argument('--sag_path', default='/ModelTest/Images/test_RS_cropped1.png', type=str)
    parser.add_argument('--trans_path', default='/ModelTest/Images/test_RT_cropped1.png', type=str)
    parser.add_argument('--age_wks', default=34, type=int)
    parser.add_argument('--left_kidney', action="store_true", help="Flag for left kidney")
    parser.add_argument('--with_cov', action="store_true", help="Flag for no Covariate model")
    # parser.add_argument('-checkpoint', default="\ModelTest\ModelWeights\SickKids_origST_TrainOnly_40epochs_bs16_lr0.001_RCFalse_covTrue_OSFalse_30thEpoch_20210614_v5.pth")
    args = parser.parse_args()

    if args.with_cov:
        checkpoint = "/ModelTest/ModelWeights/SickKids_origST_TrainOnly_40epochs_bs16_lr0.001_RCFalse_covTrue_OSFalse_30thEpoch_20210614_v5.pth"
    else:
        checkpoint = "/ModelTest/ModelWeights/NoFinalLayerFineTuneNoCov_v2_TrainOnly_40epochs_bs16_lr0.001_RCFalse_covFalse_OSFalse_30thEpoch.pth"


    dir_paste = os.getcwd().split("\\")[0:(len(os.getcwd().split("\\"))-3)]

    in_dir = "\\".join(dir_paste) + "\\"

    checkpoint = in_dir + checkpoint
    net = SiamNet(cov_layers=args.with_cov).to(device)

    if torch.cuda.is_available():
        pretrained_dict = torch.load(checkpoint)['model_state_dict']
    else:
        pretrained_dict = torch.load(checkpoint, map_location='cpu')['model_state_dict']

    model_dict = net.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(pretrained_dict)
    # for file in files_to_get:
    #     print(file)

    ## update image paths + load images
    sag_path = in_dir + args.sag_path
    trans_path = in_dir + args.trans_path

    X = load_image(sag_path, trans_path)

    ## convert images to torch float tensors
    X[0] = torch.from_numpy(X[0]).float()
    X[1] = torch.from_numpy(X[1]).float()
    combined_image = torch.unsqueeze(torch.from_numpy(X).float(), 0).to(device)

    ## run model
    net.eval()
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        net.zero_grad()
        if args.with_cov:
            output = net(combined_image, age=args.age_wks, left=args.left_kidney)
        else:
            output = net(combined_image)

        output_softmax = softmax(output)
        pred_prob = output_softmax[:, 1]


        if args.with_cov:
            ## use Platt scaling determined a priori to scale prediction
            pred_prob = np.exp(-2.311 + 3.5598*float(pred_prob))
        else:
            pred_prob = np.exp(-2.5982 + 4.318*float(pred_prob))

    print("Probability of surgery:::{:6.3f}".format(float(pred_prob)))
