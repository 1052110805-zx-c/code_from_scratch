import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
            """
            在 Python 的世界里，sublayer 就是一个普通的函数参数。
但它特殊就特殊在：它接收的不是一个数字，也不是一个张量，而是一个“干活的机器（函数或对象）”。
在执行 sublayer(...) 时，那个括号 () 就是在拉动这台机器的启动拉杆。它把洗完澡的木头 self.norm(x) 作为唯一的原材料，塞进这台机器里。
所以sublayer只能接收一个参数，就是 self.norm(x)，因为这是我们在 ResidualConnection 的 forward 方法里调用 sublayer 时传入的唯一参数。
            """

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        #因为你需要 2 台一模一样的残差连接机器！一台放在注意力后面，一台放在前馈网络后面。
        #所以不能像前面那样直接写成一个残差连接类的实例，而是需要一个 ModuleList 来存储两个残差连接实例。
        #也不能像MultiHeadAttentionBlock那样直接在构造函数里创建一个残差连接实例，因为你需要两个不同的残差连接实例，所以需要在构造函数里创建一个 ModuleList 来存储两个残差连接实例。

         """你产生了一个极其美妙但错误的错觉：“官方的机器直接写，我自己造的机器必须用 ModuleList 包起来。”
真相是：用不用 ModuleList，和“这台机器是谁造的”毫无关系！只和“你要存多少台”有绝对关系！
记住这个死理：核心只看“数量（单数还是复数）”！
问题一：
# 官方机器 (1台)
self.my_linear = nn.Linear(512, 512)

# 你自己造的机器 (1台)
self.my_res = ResidualConnection(features, dropout) # 看！直接写，不用 ModuleList！

问题二：
# 错误：大管家看不透普通的 Python 列表 []
self.many_linears = [nn.Linear(10, 10), nn.Linear(10, 10)] 
self.many_res = [ResidualConnection(512, 0.1), ResidualConnection(512, 0.1)]
因为python把列表当成一个黑盒子，里面装的是什么机器它都看不出来，所以它无法正确地管理这些机器的参数，也无法正确地把这些机器放到 GPU 上。

# 正确：只要用了列表，必须加 ModuleList
self.many_linears = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10)]) 
self.many_res = nn.ModuleList([ResidualConnection(512, 0.1), ResidualConnection(512, 0.1)])
"""
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    """
    所以这里为什么要用lambda呢？因为 self_attention_block 是一个函数（或者说一个可调用对象），它需要接受三个参数：query、key 和 value。
    而 residual_connections[0] 的 forward 方法只接受两个参数：x 和 sublayer。
    为了让 residual_connections[0] 能够调用 self_attention_block，并且传入正确的参数，我们需要使用 lambda 来创建一个匿名函数，
    这个匿名函数接受一个参数 x，然后调用 self_attention_block(x, x, x, src_mask)。
    这样，residual_connections[0] 在调用 sublayer 时，就会调用这个匿名函数，并且传入 x 作为参数，从而正确地调用 self_attention_block。
    """
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:#layer是一个ModuleList，里面装了 N 个 EncoderBlock 的实例；但是实例不是在这里创建的，而是在 build_transformer 函数里创建的。
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
"""
src_mask（源端面具）：清理包装盒里的“泡沫”
作用对象：Encoder 输入的原文（比如英文 "I love you"）。

为什么需要它？ 我们的矩阵通常是固定大小的（比如长度为 10）。如果原文只有 3 个词，剩下的 7 个空位会被填上无意义的 <PAD>（填充符）。

它的任务：告诉注意力机器：“后面那 7 个词都是用来凑数的包装泡沫，绝对不要把注意力分数分配给它们！” 它纯粹是为了防垃圾信息干扰。

2. tgt_mask（目标端面具）：考场上的“防作弊挡板”
作用对象：Decoder 正在生成的译文（比如中文 "我 爱 你"）。

极其伟大的设计：大模型生成文字是**“自回归（Autoregressive）”的，也就是一个词一个词往外蹦。
当机器正在预测第 2 个词“爱”的时候，它绝对、绝对、绝对不能看到**后面的第 3 个词“你”！
如果它看到了，那就叫“偷看答案（数据穿越 Bug）”，模型在训练时会直接满分，一到实战测试就瞬间变成白痴。
"""

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    
    """
    这里的话我们直接在 build_transformer 函数里创建所有的组件（embedding 层、位置编码层、编码器块、解码器块、投影层），然后把它们组装成一个完整的 Transformer 模型。
为什么要在 build_transformer 函数里创建这些组件，而不是在 Transformer 类的构造函数里创建呢？
这是因为我们希望 build_transformer 函数能够灵活地创建不同配置的 Transformer 模型，比如不同的层数 N、不同的头数 h、不同的隐藏层大小 d_model 等等。
如果我们把这些组件的创建放在 Transformer 类的构造函数里，那么每次我们想要创建一个新的 Transformer 模型，
我们都需要修改 Transformer 类的代码，这样就不够灵活了。
而如果我们把这些组件的创建放在 build_transformer 函数里，那么我们只需要调用 build_transformer 函数，并传入不同的参数，
就可以轻松地创建出不同配置的 Transformer模型了。


这里相当于为上述的class创造了一个工厂函数，专门用来生产Transformer模型的实例。
这个工厂函数的好处是它可以根据传入的参数灵活地创建不同配置的Transformer模型，而不需要修改Transformer类的代码。
    """
    # Create the embedding layers

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer