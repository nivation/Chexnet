# reference https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import densenet121
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        
        # get the pretrained DenseNet121 network
        self.densenet = densenet121(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features
        
        # add the average global pool
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # get the classifier of the vgg19
        self.classifier = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # don't forget the pooling
        x = self.global_avg_pool(x)
        x = x.view((1, 1024))
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)


#-------------------------------------------------------------------------------- 





# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset

database = '/home/stevenlai/Desktop/heatmap/'
dataset = datasets.ImageFolder(root=database, transform=transform)

# define the dataloader to load that image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)


# initialize and load in the pretrained model

pathModel = '/home/stevenlai/Desktop/chexnet/Full_set/model/2020_07_03-15:55:13_fullset.pth.tar'
densenet = DenseNet()
modelCheckpoint = torch.load(pathModel)
densenet.load_state_dict(modelCheckpoint['state_dict'],False)

# set the evaluation mode
densenet.eval()

# get the image from the dataloader
img, _ = next(iter(dataloader))

# get the most likely prediction of the model
pred = densenet(img).argmax(dim=1)
#print(pred)

# get the gradient of the output with respect to the parameters of the model
densenet(img)[:, 0].backward()

# pull the gradients out of the model
gradients = densenet.get_activations_gradient()

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = densenet.get_activations(img).detach()

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
    
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
#plt.matshow(heatmap.squeeze())

img = cv2.imread('/home/stevenlai/Desktop/heatmap/test/CHNCXR_0662_1.png')
heatmap = heatmap.numpy()
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./map.jpg', superimposed_img)
superimposed_img = superimposed_img/np.max(superimposed_img) *255
plt.imshow(superimposed_img/255)
plt.show()
