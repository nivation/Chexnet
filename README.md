---Files----------------------------------------------------------------------------------------

folder 'original' is the original code download from https://github.com/zoogzog/chexnet
folder MON(Can_use_for_testing) is a small testing set for the model, which doesnt get debugged for a long time
folder Full_set is the main training file
folder 'database' sort the whole data w.r.t its source, and is use for our training


---Full_set:------------------------------------------------------------------------------
the main training file

---Full_set/dataset:---------------------------------------------------------------------------
Training:Val:Test = 4:1:1
where cv = 5 within training and val data
You can set your own cv in Main.py, and found the result of txt in folder 'cv'
>>> txtfile = CrossValidation(fold = 5) ==> train:val = 4:1
the rest of the txt file is use for confusion matrix w.r.t source

---train---------------------------------------------------------------------
To Train your model, remember to switch mode in Main.py
>>>  runTest()    >>>  #runTest() 
     #runTrain()       runTrain() 
     
Change class according to your own class. In our case, it's only a binary task, so 1 class
>>> nnClassCount = 1

You can choose whether or not change the training parameter. The default is listed:
>>> trBatchSize = 640
>>> trMaxEpoch = 30

Run terminal in /home/stevenlai/Desktop/chexnet/Full_set/Main.py:
>>> python3 Main.py
It should start woring like:

preparing txtfile for cross validation...
fold: 1 complete
fold: 2 complete
fold: 3 complete
fold: 4 complete
fold: 5 complete

training for fold 1

Batch size: 640
Total epoch: 30

Train: 443730
Train Acc: 75280
Val: 18818
------- epoch: 1 ------------------------------------------------
[save] [2020_07_10-18:53:13]
train acc = 0.932  train loss = 0.454
val acc = 0.929  val loss = 0.214

epoch: 2
[save] [2020_07_10-21:02:08]
train acc = 0.927  train loss = 0.411
val acc = 0.925  val loss = 0.211 ...

---test---------------------------------------------------------------------
To test your model, remember to switch mode in Main_testwhentraining.py
>>>  #runTest()    >>>  runTest() 
     runTrain()         #runTrain() 
     
Change class according to your own class. In our case, it's only a binary task, so 1 class
>>> nnClassCount = 1

You can choose whether or not change the training parameter. The default is listed:
>>> trBatchSize = 1

Set your own model:
>>> pathModel = "your model's path.tar"

Run terminal in /home/stevenlai/Desktop/chexnet/Full_set/:
>>> python3 Main_testwhentraining.py
It should start woring like:

Testing default
Batch size: 1
model loaded: /home/stevenlai/Desktop/chexnet/Full_set/model/2020_07_20-14:52:07_fullset.pth.tar
Test: 18822
Start Testing...

and you may find confusion matrix and result csvfile output at /home/stevenlai/Desktop/chexnet/Full_set/plot/
remember to sort it in folders or it might be covered 


---heatmap--------------------------------------------------------------------
open jupyter notebook in /home/stevenlai/Desktop/chexnet/Full_set/plot:
>>> jupyter notebook

open file '/home/stevenlai/Desktop/chexnet/Full_set/heamap2.ipynb' in jupyter

























