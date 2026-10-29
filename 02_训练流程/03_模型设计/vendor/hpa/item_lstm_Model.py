import math
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置目标索引为1以及设置随机种子以确保实
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0
        self.acfuction = nn.Tanh()
        self.bn = nn.BatchNorm1d(12)
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(self.head_dim, hid_dim)
        self.fc_k = nn.Linear(self.head_dim, hid_dim)
        self.fc_v = nn.Linear(self.head_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, self.head_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        x = self.bn(x)
        x=self.acfuction(x)
        # x = [batch size, query len, hid dim]

        return x, attention


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        # 计算正弦和余弦部分
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码矩阵添加为模型的缓冲区，这样它就不会在优化过程中被更新
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码添加到输入序列
        temp=self.pe[:x.size(1), :]
        x = x + temp
        return self.dropout(x)
class Seq2SeqAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, dropout=0.1, device=device, num_layers=2):
        super(Seq2SeqAttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 在LSTM前添加的全连接层
        self.fc_encoder = nn.Linear(input_dim, hidden_dim) #编码器层只输入上游信息，为针对时滞特征提取
        # 添加基于公式的位置编码
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        temp = np.tril(np.ones((12,12)))
        self.mask = torch.from_numpy(temp).to(device)
        self.attention1 = MultiHeadAttentionLayer(hidden_dim * n_heads, n_heads, dropout, device)
        self.attention2 = MultiHeadAttentionLayer(hidden_dim*n_heads, n_heads, dropout, device)
        self.attention3 = MultiHeadAttentionLayer(hidden_dim * n_heads, n_heads, dropout, device)
        self.attention4 = MultiHeadAttentionLayer(hidden_dim * n_heads, n_heads, dropout, device)
        self.attention5 = MultiHeadAttentionLayer(hidden_dim * n_heads, n_heads, dropout, device)
        # self.attention6 = MultiHeadAttentionLayer(hidden_dim * n_heads, n_heads, dropout, device)
        self.dp = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.Tanh= nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.linear1 = nn.Linear(1, 32)
        self.linear2 = nn.Linear(2, 32)
        # self.linear3 = nn.Linear(32, 1)
        # self.linear4 = nn.Linear(32 * 2, 32)
        self.linear5 = nn.Linear(32 * 2, 1)
        # self.linear6 = nn.Linear(32, 1)
        self.max_pool_layer = nn.MaxPool1d(32)  # 对32步进行最大池化
        # self.acfun2 = nn.ReLU()
        self.bn = nn.BatchNorm1d(12)
        #self.bn1 = nn.BatchNorm1d(32)
        # self.linear6 = nn.Linear(12, 3)
        #self.w = nn.Parameter(torch.ones(3))

    def forward(self, encoder_input,decoder_input):#Qd_t36号闸的当前时刻的流量,Ht_1，上一时刻的闸前水位
        input_q = encoder_input[:,:,4]  #输入流量

        LSTM_input = encoder_input[:, :,[0,1,2,3,4]]#[0,1,2,3,4,5,6,9]
        # 通过全连接层
        encoder_input = self.fc_encoder(LSTM_input)
        # 添加位置编码
        encoder_input = self.positional_encoding(encoder_input)
        # 编码器前向传播
        attention_output, _ = self.attention1(encoder_input, encoder_input, encoder_input, self.mask)
        attention_output, _ = self.attention2(attention_output, attention_output, attention_output, self.mask)
        attention_output, _ = self.attention3(attention_output, attention_output, attention_output, self.mask)
        attention_output, _ = self.attention4(attention_output, attention_output, attention_output, self.mask)
        attention_output, _ = self.attention5(attention_output, attention_output, attention_output, self.mask)
        # attention_output, _ = self.attention6(attention_output, attention_output, attention_output, self.mask)
        #attention_output1=attention_output.reshape(attention_output.shape[0],-1)
        input_q_f = torch.unsqueeze(input_q, 2)
        input_q = self.linear1(input_q_f)
        input_q = self.Tanh(input_q)
        # 应用最大池化层
        input_q = self.max_pool_layer(input_q)
        #input_q = self.bn(input_q)
        #input_q=input_q+input_q_f
        input_q =self.softmax(input_q)
        # input_q = self.sigmoid(input_q)
        input_q = input_q.permute(0, 2, 1)
        attention_output=self.sigmoid(attention_output)
        # attention_output = self.Tanh(attention_output)
        # 对位相乘
        encoder_output =torch.bmm(input_q ,attention_output)
        # encoder_output  = input_q  * self.sigmoid(attention_output)
        # encoder_output = encoder_output.permute(0,2,1)
        # 使用 squeeze 函数移除大小为 1 的维度
        output_tensor = encoder_output.squeeze(1)
        decoder_output= self.dp(self.linear2(decoder_input))
        decoder_output = self.Tanh(decoder_output)
        #做拼接
        decoder_output = torch.cat((decoder_output,output_tensor),1)
        # decoder_output = self.linear4(decoder_output)
        # decoder_output = self.Tanh(decoder_output)
        # decoder_output = self.linear3(decoder_output)
        decoder_output = self.linear5(decoder_output)
        return decoder_output
