import torch
import math
from torch import nn
import torch.nn.functional as F


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
                    nn.Tanh(),
                    nn.Linear(2*embbed_size,embbed_size),
                    nn.Tanh())
        
        self.ffn2 =nn.Sequential(
                    nn.Linear(embbed_size,2*embbed_size),
                    nn.Tanh(),
                    nn.Linear(2*embbed_size,embbed_size),
                    nn.Tanh())
        
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


attention = InterFusion(embbed_size=6,num_hiddens=8,num_heads=3)
attention.eval()

batch_size=10
embbed_size=6

X = torch.ones((batch_size, 2, embbed_size))

print(attention(X).shape)