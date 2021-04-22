import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_size=None, embedding_matrix=None,
                 fine_tune=True, dropout=0.5,
                 padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False,
                 device=torch.device("cpu")):
        '''
        Embedding Layer need at least one of `embedding_size` and `embedding_matrix`
        :param embedding_size: tuple, contains 2 integers indicating the shape of embedding matrix, eg: (20000, 300)
        :param embedding_matrix: torch.Tensor, the pre-trained value of embedding matrix
        :param fine_tune: boolean, whether fine tune embedding matrix
        :param dropout: float, dropout rate
        :param padding_idx: int, if given, pads the output with zeros whenever it encounters the index
        :param max_norm: float, if given, will renormalize the embeddings to always have a norm lesser than this
        :param norm_type: float, the p of the p-norm to compute for the max_norm option
        :param scale_grad_by_freq: boolean, if given, this will scale gradients by the frequency of the words in the mini-batch
        :param sparse: boolean, *unclear option copied from original module*
        '''
        super(EmbeddingLayer, self).__init__()

        if embedding_matrix is not None:
            embedding_size = embedding_matrix.shape
        else:
            embedding_matrix = torch.nn.init.uniform_(torch.FloatTensor(embedding_size[0], embedding_size[1]),
                                                      a=-0.15,
                                                      b=0.15)
        assert (embedding_size is not None)
        assert (embedding_matrix is not None)
        # Config copying
        self.matrix = nn.Embedding(num_embeddings=embedding_size[0],
                                   embedding_dim=embedding_size[1],
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type,
                                   scale_grad_by_freq=scale_grad_by_freq,
                                   sparse=sparse)
        self.matrix.weight.data.copy_(embedding_matrix)
        self.matrix.weight.requires_grad = fine_tune
        self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None

        self.device = device
        self.to(device)

    def forward(self, x):
        '''
        Forward this module
        :param x: torch.LongTensor, token sequence or sentence, shape is [batch, sentence_len]
        :return: torch.FloatTensor, output data, shape is [batch, sentence_len, embedding_size]
        '''
        if self.dropout is not None:
            return F.dropout(self.matrix(x), p=self.dropout, training=self.training)
        else:
            return self.matrix(x)


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False, device=torch.device("cpu")):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.device = device
        self.to(device)

    def forward(self, x, x_len, only_use_last_hidden_state=False):
        """
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort

        :param x: FloatTensor, pre-padded input sequence (batch_size, seq_len, feature_dim)
        :param x_len: numpy list, indicating corresponding actual sequence length
        :return: output, (h_n, c_n)
        - **output**: FloatTensor, packed output sequence (batch_size, seq_len, feature_dim * num_directions)
            containing the output features `(h_t)` from the last layer of the LSTM, for each t.
        - **h_n**: FloatTensor, (num_layers * num_directions, batch, hidden_size)
            containing the hidden state for `t = seq_len`
        - **c_n**: FloatTensor, (num_layers * num_directions, batch, hidden_size)
            containing the cell state for `t = seq_len`
        """
        # 1. sort
        x_sort_idx = np.argsort(-(x_len).cpu())
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx)).to(self.device)
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx).to(self.device)]
        # 2. pack
        x_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        # 3. process using RNN
        out_pack, (ht, ct) = self.LSTM(x_p, None)
        # 4. unsort h
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)

        if only_use_last_hidden_state:
            return ht
        else:
            # 5. unpack output
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            # 6. unsort out c
            out = out[x_unsort_idx]
            ct = torch.transpose(ct, 0, 1)[x_unsort_idx]
            ct = torch.transpose(ct, 0, 1)
            return out, (ht, ct)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, edge_types, dropout=0.5, bias=True, use_bn=False,
                 device=torch.device("cpu")):
        """
        Single Layer GraphConvolution

        :param in_features: The number of incoming features
        :param out_features: The number of output features
        :param edge_types: The number of edge types in the whole graph
        :param dropout: Dropout keep rate, if not bigger than 0, 0 or None, default 0.5
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_types = edge_types
        self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None
        # parameters for gates
        self.Gates = nn.ModuleList()
        # parameters for graph convolutions
        self.GraphConv = nn.ModuleList()
        # batch norm
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

        for _ in range(edge_types):
            self.Gates.append(BottledOrthogonalLinear(in_features=in_features,
                                                      out_features=1,
                                                      bias=bias))
            self.GraphConv.append(BottledOrthogonalLinear(in_features=in_features,
                                                          out_features=out_features,
                                                          bias=bias))
        self.device = device
        self.to(device)

    def forward(self, x_text, adj):
        """

        :param x_text: FloatTensor, x_text feature tensor, (batch_size, seq_len, hidden_size)
        :param adj: FloatTensor (sparse.FloatTensor.to_dense()), adjacent matrix for provided graph of padded sequences, (batch_size, edge_types, seq_len, seq_len)
        :return: output
            - **output**: FloatTensor, output feature tensor with the same size of x_text, (batch_size, seq_len, hidden_size)
        """
        adj_ = adj.transpose(0, 1)  # (edge_types, batch_size, seq_len, seq_len)
        ts = []
        for i in range(self.edge_types):
            gate_status = torch.sigmoid(self.Gates[i](x_text))  # (batch_size, seq_len, 1)
            try:
                adj_hat_i = adj_[i] * gate_status  # (batch_size, seq_len, seq_len)
            except Exception as e:
                print(x_text.size())
                print(adj.size())
                raise e
            ts.append(torch.bmm(adj_hat_i, self.GraphConv[i](x_text)))
        ts = torch.stack(ts).sum(dim=0, keepdim=False).to(self.device)
        if self.use_bn:
            ts = ts.transpose(1, 2).contiguous()
            ts = self.bn(ts)
            ts = ts.transpose(1, 2).contiguous()
        ts = F.relu(ts)
        if self.dropout is not None:
            ts = F.dropout(ts, p=self.dropout, training=self.training)
        return ts

class HighWay(nn.Module):
    def __init__(self, size, num_layers=1, dropout_ratio=0.5):
        super(HighWay, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.trans = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = dropout_ratio

        for i in range(num_layers):
            tmptrans = BottledXavierLinear(size, size)
            tmpgate = BottledXavierLinear(size, size)
            self.trans.append(tmptrans)
            self.gate.append(tmpgate)

    def forward(self, x):
        '''
        forward this module
        :param x: torch.FloatTensor, (N, D) or (N1, N2, D)
        :return: torch.FloatTensor, (N, D) or (N1, N2, D)
        '''

        g = torch.sigmoid(self.gate[0](x))
        h = F.relu(self.trans[0](x))
        x = g * h + (1 - g) * x

        for i in range(1, self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            g = torch.sigmoid(self.gate[i](x))
            h = F.relu(self.trans[i](x))
            x = g * h + (1 - g) * x

        return x

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.5, alpha=1e-2, hidden_size=64, concat=True, device=torch.device("cpu")):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features)
        a_layers = [
            nn.Linear(2 * out_features, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.device = device
        self.to(device)

    def forward(self, x_text, adj):
        B, N = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1) # (batch_size, n*n)
        h = self.W(x_text) # (B, N, D)
        h1 = h.repeat(1, 1, N).view(B, N * N, -1)
        h2 = h.repeat(1, N, 1)

        a_input = torch.cat([h1, h2], dim=2) # (B, N*N, 2*D)
        e = self.leakyrelu(self.afcs(a_input)).squeeze(2) # (B, N*N)

        attention = F.softmax(mask_logits(e, dmask), dim=1) # (B, N*N)
        if self.dropout != None:
            attention = F.dropout(attention, self.dropout, training=self.training)
        attention = attention.view(*adj.size()) # (B, N, N)
        feature = attention.bmm(h)
        if self.dropout != None:
            feature = F.dropout(feature, self.dropout, training=self.training)

        if self.concat:
            feature = F.elu(feature)
        return feature

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, in_features, out_features, edge_types, dropout=0.5, use_bn=False, alpha=1e-2, nheads=8, device=torch.device("cpu"), concat=True):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.device = device
        self.edge_types = edge_types
        self.out_features = out_features
        self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None

        assert (out_features % nheads == 0)
        self.att_list = []
        for _ in range(edge_types):
            self.att_list.append([GraphAttentionLayer(in_features, int(out_features/nheads), dropout=self.dropout, alpha=alpha, concat=True, device=device) for _ in range(nheads)])
        self.attentions = [GraphAttentionLayer(in_features, int(out_features/nheads), dropout=self.dropout, alpha=alpha, concat=True, device=device) for _ in range(nheads)]

        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

        self.to(device)

    def forward(self, x, adj):
        adj_ = adj.transpose(0, 1) # (edge_types, batch_size, seq_len, seq_len)
        xx = []
        for i in range(self.edge_types):
            x = torch.cat([att(x, adj_[i]) for att in self.att_list[i]], dim=2) # (batch, seq_len, d')
            xx.append(x)
        xx = torch.stack(xx).sum(dim=0, keepdim=False).to(self.device) # (batch, seq_len, d')
        if self.use_bn:
            xx = xx.transpose(1, 2).contiguous()
            xx = self.bn(xx)
            xx = xx.transpose(1, 2).contiguous()
        xx = F.relu(xx)
        if self.dropout != None:
            xx = F.dropout(xx, self.dropout, training=self.training)
        return xx

class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)

class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(OrthogonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.orthogonal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class XavierLinear(nn.Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class BottledOrthogonalLinear(Bottle, OrthogonalLinear):
    pass

class BottledXavierLinear(Bottle, XavierLinear):
    pass


class GraphAttConvolution(nn.Module):
    def __init__(self, in_features, out_features, edge_types, dropout=0.5, bias=True, use_bn=False,
                 device=torch.device("cpu"), alpha=1e-2, nheads=8, concat=True):
        """
        Single Layer GraphAttConvolution

        :param in_features: The number of incoming features
        :param out_features: The number of output features
        :param edge_types: The number of edge types in the whole graph
        :param dropout: Dropout keep rate, if not bigger than 0, 0 or None, default 0.5
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        """
        super(GraphAttConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_types = edge_types
        self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None
        # parameters for gates
        self.Gates = nn.ModuleList()
        # parameters for graph convolutions
        self.GraphConv = nn.ModuleList()
        # batch norm
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

        for _ in range(edge_types-1):
            self.Gates.append(BottledOrthogonalLinear(in_features=in_features,
                                                      out_features=1,
                                                      bias=bias))
            self.GraphConv.append(BottledOrthogonalLinear(in_features=in_features,
                                                          out_features=out_features,
                                                          bias=bias))
        self.att_list = []
        for _ in range(1):
            self.att_list.append([GraphAttentionLayer(in_features, int(out_features/nheads), dropout=self.dropout, alpha=alpha, concat=True, device=device) for _ in range(nheads)])

        self.device = device
        self.to(device)

    def forward(self, x_text, adj):
        """

        :param x_text: FloatTensor, x_text feature tensor, (batch_size, seq_len, hidden_size)
        :param adj: FloatTensor (sparse.FloatTensor.to_dense()), adjacent matrix for provided graph of padded sequences, (batch_size, edge_types, seq_len, seq_len)
        :return: output
            - **output**: FloatTensor, output feature tensor with the same size of x_text, (batch_size, seq_len, hidden_size)
        """
        adj_ = adj.transpose(0, 1)  # (edge_types, batch_size, seq_len, seq_len)
        ts = []
        for i in range(self.edge_types):
            if i == 6:
                x = torch.cat([att(x_text, adj_[i]) for att in self.att_list[i-6]], dim=2)
                ts.append(x)
                continue
            gate_status = torch.sigmoid(self.Gates[i](x_text))  # (batch_size, seq_len, 1)
            adj_hat_i = adj_[i] * gate_status  # (batch_size, seq_len, seq_len)
            ts.append(torch.bmm(adj_hat_i, self.GraphConv[i](x_text)))
        ts = torch.stack(ts).sum(dim=0, keepdim=False).to(self.device)
        if self.use_bn:
            ts = ts.transpose(1, 2).contiguous()
            ts = self.bn(ts)
            ts = ts.transpose(1, 2).contiguous()
        ts = F.relu(ts)
        if self.dropout is not None:
            ts = F.dropout(ts, p=self.dropout, training=self.training)
        return ts

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, num_classes=4, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {},".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {}".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds_softmax, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        preds_softmax = F.softmax(preds_softmax, dim=-1)
        self.alpha = self.alpha.to(preds_softmax.device)
        preds_softmax = preds_softmax.view(-1,preds_softmax.size(-1))
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


