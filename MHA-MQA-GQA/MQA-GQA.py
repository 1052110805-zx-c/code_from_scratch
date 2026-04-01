import torch
import torch.nn as nn
import math

class UnifiedAttention(nn.model):
    def __init__(self,d_model:int,n_head:int,n_kv_head:int,dropout:float)-->None:
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

        def forward()