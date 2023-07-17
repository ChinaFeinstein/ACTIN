import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import pytorch_lightning as pl
from Mynet import ACTINet
from Data_process import MyOwnDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import KFold

'''

programed by Feinstein

Version: 1.2.2

latest data : 2022.12.21

'''

dataset = MyOwnDataset(root='E:/data/')

model = ACTINet()



kf = KFold(n_splits=5,shuffle=False)  # 初始化KFold

i=0
for train_index , test_index in kf.split(dataset):  # 调用split方法切分数据
    i=i+1
    if i>2:
        break

    print('train_index:%s , test_index: %s ' %(train_index,test_index))

data_train = Subset(dataset, train_index)
data_val = Subset(dataset, test_index)
early_stop_callback = EarlyStopping(monitor="t", min_delta=0.00, patience=6, verbose=False, mode="max")
checkpoint_callback = ModelCheckpoint(monitor='t',mode='max')
trainer = pl.Trainer(accelerator='gpu',callbacks = [early_stop_callback,checkpoint_callback])

if __name__ == '__main__': 
    train_loader = DataLoader(data_train,batch_size=978,shuffle=True)
    val_loader = DataLoader(data_val,batch_size=1000,shuffle=False)

    trainer.fit(model, train_loader, val_loader)
