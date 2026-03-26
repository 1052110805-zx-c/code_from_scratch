
import torch
import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):

 def __init__(self,d_model: int,h: int ,dropout: float)-> None:
   super().__init__()
   self.d_model=d_model
   self.h=h

   assert d_model%h == 0

   self.d_k=d_model // h
   self.w_q=nn.Linear(d_model,d_model,bias=False)
   self.w_k=nn.Linear(d_model,d_model,bias=False)
   self.w_v=nn.Linear(d_model,d_model,bias=False)
   self.w_o=nn.Linear(d_model,d_model,bias=False)

   self.dropout=nn.Dropout(dropout)


 def attention(query, key, value, mask, dropout:nn.Dropout):           #[batch,h,seq_len,d_k]
    d_k=query.shape[-1]
    attention_scores=(query@key.transpose(-2,-1))/math.sqrt(d_k)   #[batch,h,seq_len,seq_len]

    print(f"attention分数的矩阵：{attention_scores.shape}")
    if mask is not None:
        attention_scores.masked_fill_(mask == 0, -1e9)

    attention_scores=attention_scores.softmax(dim=-1)

    if dropout is not None:
        attention_scores = dropout(attention_scores)


    output = attention_scores @ value
    print(f"   [Attention内部] 打分矩阵乘以V，得到当前头输出: {output.shape}")
    return output, attention_scores   


 def forward(self, q, k, v, mask):
    print(f"刚进入流水线，输出q的形状：{q.shape}")
    query=self.w_q(q)   #[batch, seq_len,d_model]
    key=self.w_k(k)
    value=self.w_v(v)

    print(f"经过全连接层过后（词向量矩阵*wq之后得到）：{query.shape}")
##  [batch, seq_len, d_model]--->[batch,seq_len, h, d_K]--->[batch, h, seq_len, d_K]
    query=query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
    value=value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
    key=key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
    print(f"分割好多头后）：{query.shape}")

    x, self.attention_scores=MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

#   [batch, h, seq_len, d_K]--->[batch, seq_len, h, d_K]-->[batch,seq_len, d_model ]
    x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
    print(f"多个q拼接回去）：{x.shape}")

    final_out = self.w_o(x)
    print(f"经过最后一个线性层 w_o，最终输出形状: {final_out.shape}\n")
    return final_out

if __name__ == "__main__":
    BATCH_SIZE=2
    SEQ_LEN=4
    D_MODEL=8
    HEADS=2
    DROPOUT=0.1
    
    print("="*40)
    print(f"初始化模型：输入维度={D_MODEL},头数={HEADS}")
    print("="*40)

    mha=MultiHeadAttentionBlock(d_model=D_MODEL,h=HEADS,dropout=DROPOUT) #创造一个实例”mha“，调用类的init函数
    dummy_input = torch.rand(BATCH_SIZE, SEQ_LEN, D_MODEL)
    dummy_mask=None

    print("\n==开始前向传播(Forward)==")
    output=mha(dummy_input,dummy_input,dummy_input,dummy_mask)  #调用类的forward函数

    # 打印 w_q 机器内部那个真实的权重矩阵的形状
    print("w_q 内部权重矩阵的维度是:", mha.w_q.weight.shape)



