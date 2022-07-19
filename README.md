# Training CheXNet Model for Lung X-ray Grayscale Images    
  
## About CheXNet  
Reference code can be downloaded from [here](https://github.com/zoogzog/chexnet)  
This is a PyTorch implementation of the CheXNet algorithm for pathology detection in frontal chest X-ray images.   
More details can be found in [here](https://stanfordmlgroup.github.io/projects/chexnet/).

## Dataset (From [NIH Clinical Center](https://clinicalcenter.nih.gov/))
Database of chest X-ray images.   
Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737  
Unpack archives into separate folders  
images_001.tar.gz -> images_001  

## Usage  
### Train  

To Train your model, remember to change class according to your own class in Main.py. In our case, it's only a binary task, so
```
nnClassCount = 1
```
  
You can choose whether or not change the training parameter. The default is listed:
```
trBatchSize = 640
trMaxEpoch = 30
```

Run terminal :
```
python3 Main.py
```
It should start working like:

> preparing txtfile for cross validation...  
> fold: 1 complete  
> fold: 2 complete  
> fold: 3 complete  
> fold: 4 complete  
> fold: 5 complete  
>   
> training for fold 1  
>   
> Batch size: 640   
> Total epoch: 30  
>   
> Train: 443730  
> Train Acc: 75280  
> Val: 18818  
> ------- epoch: 1 ------------------------------------------------  
>   
> [save] [2020_07_10-18:53:13]  
> train acc = 0.932  train loss = 0.454  
> val acc = 0.929  val loss = 0.214  
>   
> epoch: 2  
> [save] [2020_07_10-21:02:08]  
> train acc = 0.927  train loss = 0.411   
> val acc = 0.925  val loss = 0.211 ...  

### Test  

To test your model, remember to change class according to your own class in Main_testwhentraining.py. In our case, it's only a binary task, so 1 class
```
nnClassCount = 1
```
You can choose whether or not change the training parameter. The default is listed:
```
trBatchSize = 1
```
Set your own model name:
```
pathModel = "your model's path.tar"
```

Run terminal:
```
python3 Main_testwhentraining.py
```
It should start woring like:

> Testing default  
> Batch size: 1  
> model loaded: your_path/model/2020_07_20-14:52:07_fullset.pth.tar  
> Test: 18822  
> Start Testing...  

and you may find confusion matrix and result csvfile output at your_path/plot/

### Heatmap  

Open jupyter notebook file heamap2.ipynb in your_path/plot/ and rerun the code
