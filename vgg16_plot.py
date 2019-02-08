import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
import os
import matplotlib.pyplot as plt


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


class VGGVisualization():
    def __init__(self, img_path, num_to_output):
        """
        initialize the class, including create the path where we output images
        :param img_path: read image from input image path
        :param num_to_output: the number of layers that we need to focus on
        """
        self.img_path = img_path
        self.pretrained_model = models.vgg16(pretrained=True).features

        # the number of features map to be outputted
        assert num_to_output >= 0 and num_to_output < len(self.pretrained_model), "invalid number!"
        self.num_to_output = num_to_output

        # create path to save feature maps
        if not os.path.exists('./feature_maps'):
            os.makedirs('./feature_maps')

        # create path to save weight images
        if not os.path.exists('./weight_images'):
            os.makedirs('./weight_images')

    def process_image(self):
        """
        process image
        :return: processed image
        """
        img = cv2.imread(self.img_path)
        img = preprocess_image(img)
        return img

    def get_feature(self):
        """
        get all the features maps at top k layers (k is num_to_output)
        :return: a list of features maps
        """
        input = self.process_image()
        print(input.shape)
        x = input
        features_to_output = []
        for index, layer in enumerate(self.pretrained_model):
            if index >= self.num_to_output:
                break
            x = layer(x)
            features_to_output.append(x)
        return features_to_output

    def get_single_feature(self):
        """
        respectively get a single feature map from different layers,
        because in each layers we have many features map and we do not need to output them all
        :return: single features maps in each layers
        """
        features_to_output = self.get_feature()
        feature_maps = []
        for feature in features_to_output:
            print(feature.shape)

            # take the feature map at index 0
            # because we have too many feature maps here (e.g. at layer 1, we got 64 feature maps)
            feature_map = feature[:, 0, :, :]
            print(feature_map.shape)
            feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2])
            print(feature_map.shape)
            feature_maps.append(feature_map)

        return feature_maps

    def feature_histogram(self, img, index):
        """
        plot histogram for a feature map, then save it
        :param img:  feature map
        :param index:  used to name the image
        :return:
        """
        plt.figure("hist")
        # we need to flatten it before plotting histogram.
        arr = img.flatten()
        n, bins, patches = plt.hist(arr, bins=256, normed=1, edgecolor='None', facecolor='red')
        plt.savefig('./feature_maps/feature_histogram%d' %index)
        plt.show()

    def show_feature_map(self):
        """
        show feature maps and plot its histogram
        :return:
        """
        # First, we need to get feature maps at top k layers
        feature_maps = self.get_single_feature()

        for i, feature_map in enumerate(feature_maps):
            feature_map = feature_map.data.numpy()
            # plot histogram of feature maps without transformation
            self.feature_histogram(feature_map, i)
            # in order to show the image, we have to transform it.
            # use sigmod to [0,1]
            feature_map = 1.0 / (1 + np.exp(-feature_map))
            # to [0,255]
            feature_map = np.round(feature_map * 255)
            # save
            cv2.imwrite('./feature_maps/feature_map_layer' +str(i+1) +'.jpg', feature_map)

    def plot_kernels_in_one_layer(self, tensor, index, num_cols=8):
        """
        plot the weights at top k layers, in CNN, weights are kernels
        :param tensor: weights
        :param index: to name the output image file at different layers
        :param num_cols: number of columns in our output figure.
        we have many kernels, so we want to range them in a figure.
        :return:
        """
        if not tensor.ndim == 4:
            raise Exception("assumes a 4D tensor")
        if not tensor.shape[-1] == 3:
            raise Exception("last dim needs to be 3 to plot")
        num_kernels = tensor.shape[0]
        num_rows = 1 + num_kernels // num_cols
        fig = plt.figure(figsize=(num_cols, num_rows))
        for i in range(tensor.shape[0]):
            ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
            ax1.imshow(tensor[i])
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig('./weight_images/kernels_in_layer%d'%index)
        plt.show()

    def plot_weight_histogram(self, current_weight, index):
        """
        plot weight histogram
        :param current_weight: current weight data
        :param index: to name the output histogram image at different layers
        :return:
        """
        # weights here is 4 dimension data, so we must use torch.histc to flatten it if we want
        # to plot a histogram of the weights
        hist_weight = torch.histc(current_weight)
        plt.hist(hist_weight.detach().numpy())
        plt.savefig('./weight_images/weight_histogram_in_layer%d' % index)
        plt.show()


    def plot_kernels(self):
        """
        weight plotting operation. It includes plotting kernel and plotting weight histogram.
        :return:
        """
        for index, layer in enumerate(self.pretrained_model):
            # we only focus on first num_to_output layers
            if index >= self.num_to_output:
                break
            # if this layer has no weight attribute, such as ReLU(inplace)
            if not hasattr(layer, 'weight'):
                print('This layer does not have weight')
                continue

            current_weight = layer.weight
            self.plot_kernels_in_one_layer(current_weight.data.numpy(), index, 8)
            self.plot_weight_histogram(current_weight, index)




if __name__ == '__main__':
    # get class
    myClass = VGGVisualization('./input_images/spider.png', 5)
    print(myClass.pretrained_model)
    myClass.show_feature_map()
    myClass.plot_kernels()



