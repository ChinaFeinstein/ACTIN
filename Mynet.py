import torch
import math
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
#from audtorch.metrics.functional import pearsonr
import pytorch_lightning as pl
from torch_geometric.nn import TopKPooling, GATv2Conv,global_mean_pool
import time
from sklearn.metrics import roc_auc_score
from torchmetrics import SpearmanCorrCoef
'''
programed by Feinstein

Version: 1.1.2

latest data : 2022.12.18

'''

class InterFusion(nn.Module):
    """多头注意力"""
    def __init__(self, embbed_size, num_hiddens,num_heads, bias=False, **kwargs):
        # emb_size是输入的embedding大小，num_hiddens是qkv向量的长度
        super(InterFusion, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.W_q = nn.Linear(embbed_size, num_hiddens*num_heads, bias=bias)
        self.W_k = nn.Linear(embbed_size, num_hiddens*num_heads, bias=bias)
        self.W_v = nn.Linear(embbed_size, num_hiddens*num_heads, bias=bias)
        self.W_o = nn.Linear(num_hiddens*num_heads, embbed_size, bias=bias)
        
        self.ffn1 =nn.Sequential(
                    nn.Linear(embbed_size,2*embbed_size),
                    nn.GELU(),
                    nn.Linear(2*embbed_size,embbed_size),
                    nn.GELU())
        
        self.ffn2 =nn.Sequential(
                    nn.Linear(embbed_size,2*embbed_size),
                    nn.GELU(),
                    nn.Linear(2*embbed_size,embbed_size),
                    nn.GELU())
        
        self.layernorm1 = nn.LayerNorm([2,embbed_size])
        self.layernorm2 = nn.LayerNorm([2,embbed_size])

    def forward(self, input):
        # input的形状(batch_size, 2, embedd_size):

        input_s=self.layernorm1(input)
        queries = transpose_qkv(self.W_q(input_s), self.num_heads)
        keys = transpose_qkv(self.W_k(input_s), self.num_heads)
        values = transpose_qkv(self.W_v(input_s), self.num_heads)

        # qkv的形状均为（batch_size, att_head, 2, h）

        att = torch.matmul(queries,keys.permute(0,1,3,2))/math.sqrt(self.num_hiddens)
        #att 的形状是（b, att_h, 2, 2）
        output =torch.matmul(F.softmax(att,dim=3),values)
        
        output_concat = output.permute(0,2,3,1)
        #out_concat的形状是（b，2，h，att_h）
        output_concat = output_concat.reshape(output_concat.shape[0],output_concat.shape[1],-1)
        res=self.W_o(output_concat)

        out = input+res

        out_s=self.layernorm2(out)

        s = torch.stack((self.ffn1(out_s[:,0,:]),self.ffn2(out_s[:,1,:])),dim=1)

        final= s+out

        return final



def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    X = X.reshape(X.shape[0], X.shape[1], -1, num_heads)

    #把num_att折叠到新的维度，以方便并行计算
    X = X.permute(0,3,1,2)

    # 最终输出的形状:(b,att-head,2,num_hidden)

    return X



class ACTINet(pl.LightningModule):
    def __init__(self,atoms=14,node_feature=256,common_dim=128,color_dim=128,b=0.011):
        super().__init__()
        self.atoms = atoms
        self.node_feature=node_feature
        self.common_dim=common_dim
        self.color_dim=color_dim
        self.b =b
        #self.spearman = SpearmanCorrCoef()

        #change node onehot into vetor

        self.node_embedding = nn.Sequential(
                                nn.Linear(self.atoms,self.node_feature),
                                nn.Tanh(),
                                nn.Linear(self.node_feature,self.node_feature),
                                nn.Tanh())

        self.conv1 = GATv2Conv(self.node_feature, int(self.node_feature),heads=3,concat=True)
                                   
        self.pool1 = TopKPooling(int(self.node_feature*3),ratio=0.8)

        self.conv2 = GATv2Conv(int(self.node_feature*3), int(self.node_feature*3),heads=3,concat=False)

        self.pool2 = TopKPooling(int(self.node_feature*3),ratio=0.8)


        self.drug = nn.Sequential(
                    nn.Linear(self.node_feature*3,self.common_dim),
                    nn.Tanh())

        #将基因和药物投影到同一向量空间

        self.gene = nn.Sequential(    
                    nn.Linear(128,self.common_dim),
                    nn.Tanh())
        
        self.interfusin = nn.Sequential(
            InterFusion(embbed_size=self.common_dim, num_hiddens= 64, num_heads= 4),
            InterFusion(embbed_size=self.common_dim, num_hiddens= 64, num_heads= 4),
            InterFusion(embbed_size=self.common_dim, num_hiddens= 64, num_heads= 4),
            InterFusion(embbed_size=self.common_dim, num_hiddens= 64, num_heads= 4),
            InterFusion(embbed_size=self.common_dim, num_hiddens= 64, num_heads= 4),
            InterFusion(embbed_size=self.common_dim, num_hiddens= 64, num_heads= 4))

        self.trans = nn.Sequential(
                    nn.Linear(self.common_dim*2,self.common_dim),
                    nn.Tanh(),
                    nn.Linear(self.common_dim,1))


    def Accuracy(self,y_pre,y_true):

        legal = (y_pre<=-0.125) + (y_pre>=0.125)

        y_pre = y_pre*legal

        acc = ((y_pre*y_true)>0).sum()/legal.sum()

        return acc

    
    def t_score(self,y_pre,y_true):

        legal = (y_pre<=-0.125) + (y_pre>=0.125)
        y_pre = y_pre*legal
        valid = (y_true<=-0.125) + (y_true>=0.125)
        y_true = y_true*valid

        A=((y_pre*y_true)>0).sum()+((y_pre+y_true)==0).sum()
        A=A/len(y_pre)
        E=((y_pre*y_true)<0).sum()
        E=E/len(y_true)

        return (A-E)*(A+E)



    def Scale(self,a):
        b1 = a<-1
        b2 = (a<=1) & (a>=-1)
        b3 = a>1

        a1 = 0.888*(torch.tanh(a+1))-0.125
        a2 = a/8
        a3 = 0.888*(torch.tanh(a-1))+0.125
    
        a1 = a1*b1
        a2 = a2*b2
        a3 = a3*b3
 

        ans = a1+a2+a3
        return ans
    
    def Auc(self,pre,true,state='up'):
        if state =='up':
            label = true>0.125
        if state =='down':
            label = true>-0.125
        pre =  pre.detach().cpu().numpy()
        label= label.detach().cpu().numpy()
        
        ans = roc_auc_score(label,pre)
        return ans
    


    def forward(self,data,bi=False):
        x , edge_index,gene,batch  = data.x, data.edge_index, data.gene, data.batch 
        #x是one-hot向量n*atoms
        #gene是128维向量
        gene = gene

        x = self.node_embedding(x)
        x = self.conv1(x,edge_index)
        x = F.leaky_relu(x,negative_slope=0.2) 
        global_feature1= global_mean_pool(x,batch)
        x,edge_index,_, batch, _, _ = self.pool1(x,edge_index,None,batch)
        x = self.conv2(x,edge_index)
        x = F.leaky_relu(x,negative_slope=0.2)
        x,edge_index,_, batch, _, _ = self.pool2(x,edge_index,None,batch)
        global_feature2= global_mean_pool(x,batch)

        global_feature = global_feature1+global_feature2
        #提取到drug的全局信息
        
        com_drug = self.drug(global_feature)
        #把drug和gene的特征投影到相同的向量空间
        com_gene = self.gene(gene.float())

        input = torch.stack((com_drug,com_gene),dim=1) 

        output = self.interfusin(input)

        feature_cat = torch.concat((output[:,0,:],output[:,1,:]),dim=1)

        res = self.trans(feature_cat)
        
        return res

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return optimizer
    
    def training_step(self,train_batch,batch_idx):
        data = train_batch

        y_true = self.Scale(data.y)
        y_pre = self.forward(data)

        loss = F.smooth_l1_loss(y_pre,y_true)

        loss = torch.abs(loss-self.b)+self.b

        error = ((y_pre-y_true).abs()).mean()
        acc = self.Accuracy(y_pre,y_true)

        #p_value = self.spearman(y_pre.view(-1),y_true.view(-1))

        up_auc = self.Auc(y_pre.view(-1),y_true.view(-1),state='up') 
        do_auc = self.Auc(y_pre.view(-1),y_true.view(-1),state='down') 
     
        self.log('train_error',error)
        self.log('train_acc',acc,prog_bar=True)
        self.log('train_auc',(up_auc+do_auc)/2)
        #self.log('train_p',p_value)
        self.log('train_loss',loss,on_epoch=True)
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        data = val_batch
        y_true = self.Scale(data.y)
        y_pre = self.forward(data)

        loss = F.smooth_l1_loss(y_pre,y_true)
        loss = torch.abs(loss-self.b)+self.b

        error = ((y_pre-y_true).abs()).mean()
        acc = self.Accuracy(y_pre,y_true)

        t=self.t_score(y_pre=y_pre,y_true=y_true)

        #p_value =  self.spearman(y_pre.view(-1),y_true.view(-1))
        up_auc = self.Auc(y_pre.view(-1),y_true.view(-1),state='up') 
        down_auc=self.Auc(y_pre.view(-1),y_true.view(-1),state='down')

        self.log('val_error',error)
        self.log('val_acc',acc,prog_bar=True)
        self.log('val_up_auc',up_auc)
        self.log('val_down_auc',down_auc)
        #self.log('val_p',p_value)
        self.log('t',t)
        self.log('val_loss',loss)
        return loss

    def predict_step(self, batch, batch_idx):

        data = batch
        y_p = self.forward(data)
        y_d = self.Scale(data.y)

        y_p = y_p.reshape(1,-1)
        y_d = y_d.reshape(1,-1)

        ans=torch.cat((y_p,y_d),0)

        return ans
    

#--------------------------------测试数据----------------------------------------
if __name__=='__main__':

    model = ACTINet().eval()

    x = torch.randn(15,14)

    y = torch.randn(1,1) 

    gene = torch.randn(1,128)

    edge_index = torch.tensor([[0,1,2,3,4,5,1,2,5,0], #起始点
                            [1,0,3,2,5,4,2,1,0,5]],dtype=torch.long)

    data = Data(x=x,y=y,edge_index=edge_index,gene=gene)

    t1=time.time()

    out =  model(data)

    t2 = time.time()
    print(out.shape)
    print('耗时: ',t2-t1,'s')