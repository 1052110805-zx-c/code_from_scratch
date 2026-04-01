import torch
import torch.nn as nn
import math

class UnifiedAttention(nn.Module):
    def __init__(self,d_model:int,n_head:int,n_kv_head:int,dropout:float) -> None:
        super(). __init__()
        self.d_model=d_model
        self.n_head=n_head
        self.n_kv_head=n_kv_head
        self.dropout=dropout

        assert d_model%n_head==0
        assert n_head%n_kv_head==0 

        self.d_k=d_model//n_head
        #计算每个q头对应几个head头
        self.n_q_per_kv=n_head//n_kv_head

        self.w_q=nn.Linear(d_model,d_model,bias=False)
        self.w_k=nn.Linear(d_model,self.d_k*n_kv_head,bias=False)
        self.w_v=nn.Linear(d_model,self.d_k*n_kv_head,bias=False)
        self.w_o=nn.Linear(d_model,d_model,bias=False)
        self.dropout=nn.Dropout(dropout)
@staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
            d_k=query.shape[-1]
            attention_scores=(query@key.transpose(-2,-1)/math.sqrt(d_k))
            if mask is not None:
                attention_scores=attention_scores.masked_fill_(mask==0,-1e9)
            attention_scores=attention_scores.softmax(dim=-1)

            if dropout is not None:
                attention_scores=dropout(attention_scores)

            output=attention_scores@value
            return output,attention_scores

    def forward(self,q,k,v,mask):
            query=self.w_q(q)
            key=self.w_k(k)
            value=self.w_v(v)

            #切分多头，维度变换：[batch len d_model]-->[batch len n_head d_k]-->[batch n_head len d_k]
            query=query.view(query.shape[0],query.shape[1],self.n_head,self.d_k).transpose(1,2)
            key=key.view(key.shape[0],key.shape[1],self.n_kv_head,self.d_k).transpose(1,2)
            value=value.view(value.shape[0],value.shape[1],self.n_kv_head,self.d_k).transpose(1,2)
            #如果每个q头对应的kv头大于1；就需要把kv矩阵进行重复填充；确保传入attention的qkv矩阵的维度是匹配的，是可以计算的
            #比如：如果n_head=8，n_kv_head=2，那么每个kv头就对应4个q头；所以就需要把kv矩阵在头的维度上进行重复填充4倍；这样才能和q矩阵的维度匹配
            #这里要用repeat——interleave函数；不能用repeat函数；因为repeat函数会在所有维度上进行重复；而我们只想在头的维度上进行重复；
            #所以要用repeat_interleave函数；并且指定dim=1；表示在头的维度上进行重复
            if self.n_q_per_kv>1:
                key=key.repeat_interleave(self.n_q_per_kv,dim=1)
                value=value.repeat_interleave(self.n_q_per_kv,dim=1)

            #传入attention函数
            x,self.attention_scores=UnifiedAttention.attention(query,key,value,mask,self.dropout)

            #合并多头，维度变换：[batch n_head len d_k]-->[batch len n_head d_k]-->[batch len d_model]
            x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.n_head*self.d_k)

            output=self.w_o(x)
            return output