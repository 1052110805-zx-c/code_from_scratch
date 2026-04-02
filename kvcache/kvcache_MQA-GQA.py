import torch
import torch.nn as nn
import math
from typing import Optional, Any

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

    def forward(self,q,k,v,mask,layer_ind:int=0,kv_cache:Optional[Any]=None,start_pos:int=0):#Optional的意思是，你不传入kvcache的赋值，我都有任意的复制，你不传入kvchche；那就默认为none；
           """
           “待会开工的时候，你可以选择性地（Optional）递给我一个任何款式（Any）的笔记本（kv_cache）。
           当然，如果你嫌麻烦什么都不递给我，我就默认你没带（= None），那我就不记笔记，直接硬算！”
           """
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


            #kv_cache要放在repeat_interleave之前
            if kv_cache is not None:
                # kv_cache.update 会将当前的 key 和 value 追加到缓存中，并返回拼接了历史记录的完整 key 和 value
                key,value=kv_cache.update(layer_idx,start_pos,key,value)
                #推理的时候；按照一个token的key，value传入cache是如何实现的；本质是由于外层的主程序，它不会把整段文字给到mqa；gqa；而是自回归式的；只给出当前的最新的一个字；然后计算着一个字的qkv；再把kv拼接到kvcache；在计算注意力的得分；以及向量矩阵


            if self.n_q_per_kv>1:
                key=key.repeat_interleave(self.n_q_per_kv,dim=1)
                value=value.repeat_interleave(self.n_q_per_kv,dim=1)

            #传入attention函数
            x,self.attention_scores=UnifiedAttention.attention(query,key,value,mask,self.dropout)

            #合并多头，维度变换：[batch n_head len d_k]-->[batch len n_head d_k]-->[batch len d_model]
            x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.n_head*self.d_k)

            output=self.w_o(x)
            return output