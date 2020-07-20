import os
import numpy as np
import random

class CrossValidation ():
    def __init__(self,fold = 5):
        # To insure that TB/non-TB be trained equally by different source,
        # we need to separate the data in respect to its source (Chest/Mon/Chn)
        self.fold = fold
        Chest_TB     = [] # 0
        Chest_non_TB = [] # 112120
        Mon_TB       = [] # 58
        Mon_nor      = [] # 80
        China_TB     = [] # 336
        China_nor    = [] # 326
        self.trainlist    = []
        self.vallist      = []
        self.testlist     = []

        non_TB = open('/home/stevenlai/Desktop/chexnet/database/readme/clean/N.txt','r')
        TB = open('/home/stevenlai/Desktop/chexnet/database/readme/clean/TB.txt','r')

        # Save in different list
        for name in non_TB:
            if (name.split('/')[0]) == 'MON':
                Mon_nor.append(name)
            elif (name.split('/')[0]) == 'China':
                China_nor.append(name)
            else:
                Chest_non_TB.append(name)
        #print(len(Chest_non_TB),len(Mon_nor),len(China_nor))



        for name in TB:
            if (name.split('/')[0]) == 'MON':
                Mon_TB.append(name)
            elif (name.split('/')[0]) == 'China':
                China_TB.append(name)
            else:
                Chest_TB.append(name)
        #print(len(Chest_TB),len(Mon_TB),len(China_TB))

        non_TB.close()
        TB.close()
        
        # Shuffle the list
        random.seed(10)  
        random.shuffle(Mon_nor)
        random.shuffle(Mon_TB)
        random.shuffle(China_nor)
        random.shuffle(China_TB)
        random.shuffle(Chest_non_TB)
        random.shuffle(Chest_TB)
        
        
        # Ramdomly split into different set
        # train:val:test = 4:1:1 (set val and test the same)
        train = open('/home/stevenlai/Desktop/chexnet/database/readme/clean/Train.txt','w')
        val   = open('/home/stevenlai/Desktop/chexnet/database/readme/clean/Val.txt','w')
        test   = open('/home/stevenlai/Desktop/chexnet/database/readme/clean/Test.txt','w')

        a = [Chest_TB,Chest_non_TB,China_TB,China_nor,Mon_TB,Mon_nor]

        for j in range(self.fold ):
            train_count = 0
            val_count   = 0
            test_count  = 0
            
            train_path = '/home/stevenlai/Desktop/chexnet/Full_set/dataset/cv/Train_'+'fold_'+str(j+1)+'.txt'
            val_path   = '/home/stevenlai/Desktop/chexnet/Full_set/dataset/cv/Val_'+'fold_'+str(j+1)+'.txt'
            test_path  = '/home/stevenlai/Desktop/chexnet/Full_set/dataset/cv/Test_CV_fold_'+str(fold)+'.txt'
            
            self.trainlist.append(train_path)
            self.vallist.append(val_path)
    
            train = open(train_path,'w')
            val   = open(val_path ,'w')
            test  = open(test_path ,'w')
            for cat in a:
                for i in range(len(cat)):
                    if i >= np.floor(len(cat)*fold/(fold+1)):
                        test.write(cat[i])
                        test_count += 1
                    else:
                
                        if np.floor(len(cat)/(fold+1)*j) <= i <np.floor(len(cat)/(fold+1)*(j+1)):
                            val.write(cat[i])
                            val_count += 1
                        else:
                            train.write(cat[i])
                            train_count += 1
            print('fold:',j+1,'complete')
            # print('train_count:',train_count)
            # print('val_count:',val_count)
            # print('test_count:',test_count,'(fixed)')

            
            
            train.close()
            val.close()
            test.close()
        self.testlist.append(test_path)       
        
if __name__ == '__main__':
    a = CrossValidation(fold = 5)
    # print(a.testlist)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
