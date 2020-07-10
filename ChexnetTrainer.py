import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from DatasetGenerator import DatasetGenerator
from DatasetGenerator_Imbalance import DatasetGeneratorforTraining

import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------- 

class ChexnetTrainer ():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(14, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        
        modelCheckpoint = torch.load('/home/stevenlai/Desktop/chexnet/original/models/m-25012018-123527.pth.tar')
        model.load_state_dict(modelCheckpoint['state_dict'],False)
        for param in model.parameters():
            param.requires_grad = False
            
        model.densenet121.classifier = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())
        #print(model)
        model = torch.nn.DataParallel(model).to('cuda' if torch.cuda.is_available() else 'cpu')
                
        #-------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGeneratorforTraining(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetTrainAcc = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
        print('Train:',len(datasetTrain))
        print('TrainAcc:',len(datasetTrainAcc))
        print('Val:',len(datasetVal))
        
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
        dataLoaderTrainAcc = DataLoader(dataset=datasetTrainAcc, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
        
        

        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min', verbose = 1)
                
        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(reduction = 'mean')
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        trainacclist = []
        trainlosslist = []
        vallosslist = []
        valacclist = []
        
        for epochID in range (0, trMaxEpoch):
            print('epoch:', epochID+1)
            timestampTime = time.strftime("%H:%M:%S")
            timestampDate = time.strftime("%Y_%m_%d")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            lossTrain, accTrain = ChexnetTrainer.epochTrain(model, dataLoaderTrain, dataLoaderTrainAcc, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            trainlosslist.append(lossTrain)
            trainacclist.append(accTrain)
            
            lossVal, accVal = ChexnetTrainer.epochVal(model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            vallosslist.append(lossVal)
            valacclist.append(accVal)
            
            timestampTime = time.strftime("%H:%M:%S")
            timestampDate = time.strftime("%Y_%m_%d")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(lossVal)
            
            
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()},
                           '/home/stevenlai/Desktop/chexnet/Full_set/model/' + launchTimestamp +'_fullset.pth.tar')
                print ('[save] [' + timestampEND + ']')
                print ('train acc = %.3f'%accTrain ,' train loss = %.3f'%float(lossTrain))
                print ('val acc = %.3f'%accVal , ' val loss = %.3f'%float(lossVal))
                print()
            else:
                print ( '[' + timestampEND + ']')
                print ('train acc = %.3f'%accTrain ,' train loss = %.3f'%float(lossTrain))
                print ('val acc = %.3f'%accVal ,' val loss = %.3f'%float(lossVal))
                print()
        
        plt.plot(trainacclist,label = 'training acc')
        plt.plot(valacclist,label = 'val acc')
        plt.legend(loc = 'lower right')
        plt.title('acc')
        plt.savefig('/home/stevenlai/Desktop/chexnet/Full_set/plot/'+ launchTimestamp + '_acc_fullset.png')
        plt.close()
        
        plt.plot(trainlosslist,label='training loss')
        plt.plot(vallosslist,label = 'val loss')
        plt.legend(loc = 'upper right')
        plt.title('loss')
        plt.savefig('/home/stevenlai/Desktop/chexnet/Full_set/plot/'+ launchTimestamp + '_loss_fullset.png')
        plt.close()
        
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader,dataLoaderAcc, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        
        loss_add = 0
        lossNorm = 0
        for batchID, (input, target) in enumerate (dataLoader):
                        
            target = target.cuda(non_blocking = True)
                 
            varInput = input.to('cuda' if torch.cuda.is_available() else 'cpu')
            varTarget = target.to('cuda' if torch.cuda.is_available() else 'cpu')        
            varOutput = model(varInput)
            
            lossvalue = loss(varOutput, varTarget)
            loss_add += lossvalue.item()     
            lossNorm += 1
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
        outloss = loss_add/lossNorm
    #-------------------------------------------------------------------------------- 
        
        CLASS_NAMES = [ 'Non TB', 'TB']
        cudnn.benchmark = True
        outGT = torch.FloatTensor().to('cuda' if torch.cuda.is_available() else 'cpu')
        #outPRED = torch.FloatTensor().cuda()

        correct_count = 0
        total_count = 0
        model.eval()
        
        with torch.no_grad():
            for batchID, (input, target) in enumerate (dataLoaderAcc):
                target = target.to('cuda' if torch.cuda.is_available() else 'cpu')
                outGT = torch.cat((outGT, target), 0)
                bs, c, h, w = input.size()

                varInput = (input.view(-1, c, h, w).cuda()).to('cuda' if torch.cuda.is_available() else 'cpu')
                out = model(varInput)
                
                for k in range(len(out)):
                    if out[k][0]<0.5: out[k][0] = 0.
                    else: out[k][0] = 1.
                    if out[k][0] == target[k][0]:correct_count+=1
                        
                    total_count += 1
                
        acc = correct_count/total_count
        return outloss, acc
    #-------------------------------------------------------------------------------- 
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        CLASS_NAMES = [ 'Normal', 'Abnormal']
        cudnn.benchmark = True
        outGT = torch.FloatTensor().to('cuda' if torch.cuda.is_available() else 'cpu')
        #outPRED = torch.FloatTensor().to('cuda' if torch.cuda.is_available() else 'cpu')
        correct_count = 0
        total_count = 0
        
        losstensorMean = 0
        with torch.no_grad():

            for i, (input, target) in enumerate (dataLoader):

                target = target.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking = True)

                varInput = input.to('cuda' if torch.cuda.is_available() else 'cpu')
                varTarget = target.to('cuda' if torch.cuda.is_available() else 'cpu')
                varOutput = model(varInput)

                losstensor = loss(varOutput, varTarget)

                lossVal += losstensor.item()
                lossValNorm += 1
#-------------------------------------------------------------------------------- 
                
                outGT = torch.cat((outGT, target), 0)
                bs, c, h, w = input.size()

                varInput = (input.view(-1, c, h, w).cuda()).to('cuda' if torch.cuda.is_available() else 'cpu')
                out = model(varInput)
                
                for k in range(len(out)):
                    if out[k][0]<0.5: out[k][0] = 0.
                    else: out[k][0] = 1.
                    if out[k][0] == target[k][0]:correct_count+=1
                        
                    total_count += 1
                
        outLoss = lossVal / lossValNorm
        acc = correct_count/total_count        
        return outLoss, acc
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        CLASS_NAMES = ['non-TB', 'TB']

        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        #print('classCount:',classCount)
        classCount -= 1
        for i in range(classCount):
            # outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            try:
                #print('class:',CLASS_NAMES[i])
                #print('datanpGT[:, i]',datanpGT[:, i])
                #print('datanpPRED[:, i]',datanpPRED[:, i])
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                #print('Value Error:',CLASS_NAMES[i])
                pass

        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        CLASS_NAMES = [ 'Normal', 'Abnormal']

        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'],False)
        print('model loaded')

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        model.eval()
        a = np.array([[0,0],[0,0]])
        
        print('Start Testing...')
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):
                print('epoch:',i,'out of')
                # print('input',input.size()) (1,10,3,224,224)
                # print('target',target.size()) (1,1)
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
                bs, n_crops, c, h, w = input.size()
                varInput = (input.view(-1, c, h, w).cuda()).to('cuda')
                out = model(varInput)
                # print(out.size()) (10,1)
                outMean = sum(out)/10
                outPRED = torch.cat((outPRED, outMean.data), 0)
                # print('outMean',outMean)
                # print('outPRED', outPRED)
                # Suppose they will be the same
                
                if outMean < 0.5: outMean = 0.
                else: outMean = 1.
                # print('outMean',outMean)    
                # print('target', float(target[0][0]))
                # Suppose they will be the same type
                if outMean == 1. and float(target[0][0]) == 1.: a[0][0]+=1
                if outMean == 1. and float(target[0][0]) == 0.: a[0][1]+=1
                if outMean == 0. and float(target[0][0]) == 1.: 
                    a[1][0]+=1
                    print('TB mistaken into non TB')
                if outMean == 0. and float(target[0][0]) == 0.: 
                    a[1][1]+=1
                    print('non TB mistaken into TB')
                    
        
        total_correct = a[0][0] + a[1][1]
        total = a[0][0] + a[0][1] + a[1][0] + a[1][1]
        acc = total_correct/total
        print('test acc = %.3f'%acc)
        
        # Matrix
        fig, ax = plt.subplots(figsize=(5,5))
        a = a.T
        ax.matshow(a.T, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(2):
            for j in range(2):
                ax.text(i, j, s = a[i, j], va='center', ha='center')
        plt.title('TB-nonTB matrix')
        plt.xlabel('ground truth')        
        plt.ylabel('predict')
        labels = ['TB','non-TB']
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.savefig('/home/stevenlai/Desktop/chexnet/Full_set/plot/matrix_fullset_'+launchTimeStamp+'.png')
        plt.close()
        #plt.show()
        
        
        #aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        #aurocMean = np.array(aurocIndividual).mean()
        
        #print ('AUROC mean ', aurocMean)
        
        #for i in range (0, len(aurocIndividual)):
        #    print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
     
        return acc
#-------------------------------------------------------------------------------- 





