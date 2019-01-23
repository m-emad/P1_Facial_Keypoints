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
        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 3)         # 224 - 5 + 1 = 220
        self.conv1_bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)            # (110, 110)

        self.conv2 = nn.Conv2d(32, 64, 3)      # 110 - 3 + 1 = 108
        self.conv2_bn = nn.BatchNorm2d(64)
        # after max pooling (54, 54)

        self.conv3 = nn.Conv2d(64, 128, 3)     # 54 - 3 + 1 = 52
        self.conv3_bn = nn.BatchNorm2d(128)
        # after max pooling (26, 26)
        
        
        self.conv4 = nn.Conv2d(128, 256, 3)     # 26 - 3 + 1 = 24
        self.conv4_bn = nn.BatchNorm2d(256)
        # after max pooling (12, 12)
        
        self.conv5 = nn.Conv2d(256, 512, 3)     # 12 - 3 + 1 = 10
        self.conv5_bn = nn.BatchNorm2d(512)
        # after max pooling (5, 5)
        
        self.fc1 = nn.Linear(512 * 5 * 5, 1024)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 136)
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
       
        x = self.pool((F.relu(self.conv1_bn(self.conv1(x)))))
        x = self.pool((F.relu(self.conv2_bn(self.conv2(x)))))
        x = self.pool((F.relu(self.conv3_bn(self.conv3(x)))))
        x = self.pool((F.relu(self.conv4_bn(self.conv4(x)))))
        x = self.pool((F.relu(self.conv5_bn(self.conv5(x)))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)

        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
