import os
import numpy as np
import time
import sys
import torch

from ChexnetTrainer import ChexnetTrainer

# -------------------------------------------------------------------------------- 

def main ():
    
    #runTest()
    runTrain()

# --------------------------------------------------------------------------------   

def runTrain():
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    
    timestampTime = time.strftime("%H:%M:%S")
    timestampDate = time.strftime("%Y_%m_%d")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = '/home/stevenlai/Desktop/chexnet/database'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = '/home/stevenlai/Desktop/chexnet/Full_set/dataset/Train.txt'
    pathFileVal = '/home/stevenlai/Desktop/chexnet/Full_set/dataset/Val.txt'
    pathFileTest = '/home/stevenlai/Desktop/chexnet/Full_set/dataset/Val.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 1
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 32
    trMaxEpoch = 100
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
        
    pathModel = '/home/stevenlai/Desktop/chexnet/Full_set/model/' + timestampLaunch + '_fullset.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    
    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

# -------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = './database'
    pathFileTest = './dataset/CHN/test_CHN.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 1
    trBatchSize = 32
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = './model/2020_05_29-16:12:54.pth.tar'
    
    timestampLaunch = ''
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

# -------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





