## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 5)         # 224 - 5 + 1 = 220
        self.conv1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)            # (110, 110)

        self.conv2 = nn.Conv2d(32, 64, 3)      # 110 - 3 + 1 = 108
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)           # (54, 54)

        self.conv3 = nn.Conv2d(64, 128, 3)     # 54 - 3 + 1 = 52
        self.conv3_bn = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)           # (26, 26)

        self.fc1 = nn.Linear(128 * 26 * 26, 1000)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1000, 136)
        

        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
       
        x = self.pool1((F.relu(self.conv1_bn(self.conv1(x)))))
        x = self.pool2((F.relu(self.conv2_bn(self.conv2(x)))))
        x = self.pool3((F.relu(self.conv3_bn(self.conv3(x)))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)

        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
