import torch
import torch.nn.functional as F
from Mynet import ACTINet
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from predata_process import MyPreDataset
import pandas as pd
import numpy as np


diff_num=483
#加载不同的训练好的模型
file = input("file path: ")
model = ACTINet.load_from_checkpoint(str(file))
id=input('iter: ')
id = str(id)
data_drug = pd.read_excel('c:/Users/Fenistein/Desktop/drug'+id+'.xlsx')
#加载预测的数据集

data_pre= MyPreDataset('E:/data')

test_loader = DataLoader(data_pre,batch_size=900,shuffle=False)
trainer = pl.Trainer(accelerator='gpu')

#predictions 是包含tensor的列表
predictions = trainer.predict(model, dataloaders=test_loader)
#把列表转化为一个tensor
all=torch.cat([x.float() for x in predictions], dim=1)
all = all.reshape(2,diff_num,-1)



def score(A,E):
    x2=A/(1+A-E)

    i=1*((A-E)>0)-1*((A-E)<0)
    
    s=200*i*torch.sqrt((A-x2)**2+(E-x2)**2)
    return s


input1=all[1,:,:].squeeze(0)  #true
input0=all[0,:,:].squeeze(0)  #pre
#得到的input shape是[diff_num,drug_num]

p=input0  #预测的结果
ill=input1  #疾病的表达

vaild=(p>=0.125)+(p<=-0.125)
mat=vaild*p*ill
#mat 的shape [gene,num_drug]

a=mat<-1e-5  #有治疗基因的比例
a=torch.sum(a,dim=0)/mat.shape[0]

e=mat>1e-5  #有加重基因的比例
e=torch.sum(e,dim=0)/mat.shape[0]

s=score(a,e)

out=s.reshape(-1,1).numpy()
a= a.reshape(-1,1).numpy()
e= e.reshape(-1,1).numpy()

res = np.concatenate([out,a,e],axis=1)


idx = data_drug['name'].to_list()#搞到药名的列表

if len(idx)==out.shape[0]:
    data_df = pd.DataFrame(data=res,index=idx,columns=['score','A','E'])
    data_df.to_csv('E:/pred/'+file[10:22]+id+'.csv')
   
    print("预测药物数量：",len(idx))
    print("文件已保存！"+file[10:22])

else:
    print("预测药物数量不匹配，检查数据。")
    print('药物数量：',len(idx))
    print('预测数量：',out.shape[0])




