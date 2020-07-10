import os
import numpy as np
import time
import sys
import torch

from ChexnetTrainer import ChexnetTrainer
from Cross_Validation import CrossValidation
# -------------------------------------------------------------------------------- 

def main ():
    
    #runTest()
    runTrain()

# --------------------------------------------------------------------------------   

def runTrain():
    #---- Generate 5-fold txt for cross-validation
    print('preparing txtfile for cross validation...')
    txtfile = CrossValidation(fold = 5)
    train = txtfile.trainlist
    val   = txtfile.vallist
    test  = txtfile.testlist
    
    
    for i in range(len(train)): 
        print()
        print('training for fold',(i+1))
        print()
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
        pathFileTrain = train[i]
        pathFileVal = val[i]
        pathFileTest = test[0]
    
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
    
        
        ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    
    
        ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

# -------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = '/home/stevenlai/Desktop/chexnet/database'
    pathFileTest = '/home/stevenlai/Desktop/chexnet/Full_set/dataset/CHN_TB.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 1
    trBatchSize = 1
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = '/home/stevenlai/Desktop/chexnet/Full_set/model/2020_07_03-15:55:13_fullset.pth.tar'
    
    timestampLaunch = ''
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

# -------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





