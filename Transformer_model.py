"""
文件说明：
本模块实现了基于Transformer架构的机器翻译模型，包含完整的编码器-解码器结构。
主要组件包括：
1. 位置编码（PositionalEncoding）
2. 词嵌入层（Embedding）
3. 层归一化（LayerNorm）
4. 前馈网络（FeedForward）
5. 多头注意力机制（MultiHeadAttention）
6. 编码器层（EncoderLayer）和解码器层（DecoderLayer）
7. 完整的编码器（Encoder）和解码器（Decoder）
8. 最终的Transformer模型组合

模型特点：
- 遵循《Attention Is All You Need》论文架构
- 支持自定义配置参数（通过config.TransformerConfig类）
- 使用Xavier初始化权重
- 包含梯度裁剪和学习率调度支持
- 符合PyTorch的模块化设计规范
"""

import torch
from torch import nn
import torch.nn.functional as F
import math

def initialize_weights(module):
    """
    初始化网络权重
    - 线性层使用Xavier均匀初始化
    - 嵌入层使用Xavier均匀初始化
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    - 解决序列数据的位置信息问题
    - 使用正弦和余弦函数生成固定位置编码
    """
    def __init__(self, config):
        super().__init__()
        # 初始化位置编码矩阵
        self.encoding = torch.zeros(config.max_len, config.d_model, device=config.device)
        self.pos = torch.arange(0, config.max_len, device=config.device).unsqueeze(1)  # 位置索引
        self.div = torch.arange(0, config.d_model, 2, device=config.device)  # 维度划分

        # 计算位置编码
        self.encoding[:, 0::2] = torch.sin(self.pos / (10000 ** (self.div / config.d_model)))
        self.encoding[:, 1::2] = torch.cos(self.pos / (10000 ** (self.div / config.d_model)))
        
        # 注册为buffer，确保模型保存时包含该参数
        self.register_buffer("pe", self.encoding)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :return: 带位置编码的张量
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].to(x.device)


class Embedding(nn.Module):
    """
    词嵌入层
    - 使用nn.Embedding实现
    - 对输出进行缩放（除以sqrt(d_model)）
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model).to(config.device)
        self.d_model = config.d_model

    def forward(self, x):
        """
        前向传播
        :param x: 输入索引张量 (batch_size, seq_len)
        :return: 嵌入张量 (batch_size, seq_len, d_model)
        """
        d_model_tensor = torch.tensor(self.d_model, dtype=torch.float32, device=x.device)
        return self.embedding(x) / torch.sqrt(d_model_tensor)


class LayerNorm(nn.Module):
    """
    层归一化模块
    - 独立于batch的归一化操作
    - 包含可学习的缩放和平移参数
    """
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.d_model, device=config.device))
        self.bias = nn.Parameter(torch.zeros(config.d_model, device=config.device))

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :return: 归一化后的张量
        """
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class FeedForward(nn.Module):
    """
    前馈神经网络模块
    - 包含两个线性层和ReLU激活函数
    - 使用Dropout防止过拟合
    """
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff).to(config.device)
        self.linear2 = nn.Linear(config.d_ff, config.d_model).to(config.device)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        initialize_weights(self.linear1)
        initialize_weights(self.linear2)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :return: 输出张量 (batch_size, seq_len, d_model)
        """
        return self.dropout2(self.linear2(self.dropout1(torch.relu(self.linear1(x)))))


class Attention(nn.Module):
    """
    缩放点积注意力模块
    - 实现基本的注意力计算
    - 包含Softmax和Dropout
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.d_k = config.d_model // config.n_head  # 每个头的维度
        self.d_model = config.d_model

    def forward(self, query, key, value, mask=None):
        """
        前向传播
        :param query: 查询张量 (batch_size, n_head, seq_len, d_k)
        :param key: 键张量 (batch_size, n_head, seq_len, d_k)
        :param value: 值张量 (batch_size, n_head, seq_len, d_k)
        :param mask: 掩码张量 (batch_size, 1, 1, seq_len)
        :return: 注意力输出 (batch_size, n_head, seq_len, d_k)
        """
        d_k_tensor = torch.tensor(self.d_k, dtype=torch.float32, device=query.device)
        score = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k_tensor)
        
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # 屏蔽无效位置
        
        score = self.softmax(score)
        score = self.dropout(score)
        return torch.matmul(score, value)


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块
    - 将输入拆分为多个头并行计算注意力
    - 合并结果后通过线性层输出
    """
    def __init__(self, config):
        super().__init__()
        self.d_k = config.d_model // config.n_head
        self.d_model = config.d_model
        self.n_head = config.n_head
        
        # 线性变换层
        self.w_k = nn.Linear(config.d_model, config.d_model).to(config.device)
        self.w_q = nn.Linear(config.d_model, config.d_model).to(config.device)
        self.w_v = nn.Linear(config.d_model, config.d_model).to(config.device)
        
        self.attention = Attention(config)
        self.linear = nn.Linear(config.d_model, config.d_model).to(config.device)
        self.dropout = nn.Dropout(config.dropout)
        initialize_weights(self.w_k)
        initialize_weights(self.w_q)
        initialize_weights(self.w_v)
        initialize_weights(self.linear)

    def forward(self, query, key, value, mask=None):
        """
        前向传播
        :param query: 查询张量 (batch_size, seq_len, d_model)
        :param key: 键张量 (batch_size, seq_len, d_model)
        :param value: 值张量 (batch_size, seq_len, d_model)
        :param mask: 掩码张量 (batch_size, 1, 1, seq_len)
        :return: 输出张量 (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # 线性变换并拆分为多头
        query = self.w_q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        
        # 计算注意力
        x = self.attention(query, key, value, mask)
        
        # 合并多头并通过线性层
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dropout(self.linear(x))


class EncoderLayer(nn.Module):
    """
    编码器层模块
    - 包含多头自注意力和前馈网络
    - 使用残差连接和层归一化
    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = LayerNorm(config)
        self.layer_norm2 = LayerNorm(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :param mask: 源序列掩码 (batch_size, 1, 1, seq_len)
        :return: 输出张量 (batch_size, seq_len, d_model)
        """
        # 自注意力子层
        norm_x = self.layer_norm1(x)
        attn_output = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout1(attn_output)  # Add & Dropout

        # 前馈网络子层
        norm_x = self.layer_norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_output)  # Add & Dropout

        return x


class DecoderLayer(nn.Module):
    """
    解码器层模块
    - 包含两个多头注意力（自注意力和编-解码注意力）
    - 使用残差连接和层归一化
    """
    def __init__(self, config):
        super().__init__()
        self.attention1 = MultiHeadAttention(config)  # 自注意力
        self.attention2 = MultiHeadAttention(config)  # 编-解码注意力
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = LayerNorm(config)
        self.layer_norm2 = LayerNorm(config)
        self.layer_norm3 = LayerNorm(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :param encoder_output: 编码器输出 (batch_size, src_len, d_model)
        :param src_mask: 源序列掩码 (batch_size, 1, 1, src_len)
        :param tgt_mask: 目标序列掩码 (batch_size, 1, 1, tgt_len)
        :return: 输出张量 (batch_size, seq_len, d_model)
        """
        # 自注意力子层
        norm_x = self.layer_norm1(x)
        attn_output1 = self.attention1(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout1(attn_output1)  # Add & Dropout

        # 编-解码注意力子层
        norm_x = self.layer_norm2(x)
        attn_output2 = self.attention2(norm_x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output2)  # Add & Dropout

        # 前馈网络子层
        norm_x = self.layer_norm3(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout3(ff_output)  # Add & Dropout

        return x


class Encoder(nn.Module):
    """
    编码器模块
    - 包含嵌入层、位置编码和多个编码器层
    - 末尾添加层归一化
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.positional_encoding = PositionalEncoding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])
        self.layer_norm = LayerNorm(config)  # 论文中编码器末尾有LayerNorm

    def forward(self, x, mask=None):
        """
        前向传播
        :param x: 输入索引张量 (batch_size, seq_len)
        :param mask: 源序列掩码 (batch_size, 1, 1, seq_len)
        :return: 编码器输出 (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.layer_norm(x)  # 应用最终层归一化
        return x


class Decoder(nn.Module):
    """
    解码器模块
    - 包含嵌入层、位置编码和多个解码器层
    - 末尾添加层归一化
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.positional_encoding = PositionalEncoding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
        self.layer_norm = LayerNorm(config)  # 论文中解码器末尾有LayerNorm

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        :param x: 输入索引张量 (batch_size, seq_len)
        :param encoder_output: 编码器输出 (batch_size, src_len, d_model)
        :param src_mask: 源序列掩码 (batch_size, 1, 1, src_len)
        :param tgt_mask: 目标序列掩码 (batch_size, 1, 1, tgt_len)
        :return: 解码器输出 (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        x = self.layer_norm(x)  # 应用最终层归一化
        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型
    - 组合编码器和解码器
    - 包含最终的线性输出层
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        :param src: 源序列索引 (batch_size, src_len)
        :param tgt: 目标序列索引 (batch_size, tgt_len)
        :param src_mask: 源序列掩码 (batch_size, 1, 1, src_len)
        :param tgt_mask: 目标序列掩码 (batch_size, 1, 1, tgt_len)
        :return: 预测对数概率 (batch_size, tgt_len, vocab_size)
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.linear(decoder_output)
        return output