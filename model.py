import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from module import EmbeddingLayer, DynamicLSTM, GraphConvolution, HighWay, GAT, BottledXavierLinear, GraphAttConvolution, FocalLoss

class Aspect_CS_GAT_BERT(nn.Module):
    """docstring for Aspect_CS_GAT"""
    def __init__(self, args):
        super(Aspect_CS_GAT_BERT, self).__init__()
        self.args = args
        self.wembeddings = args.bert_model

        # POS-Tagging Embedding Layer
        self.pembeddings = EmbeddingLayer(embedding_size=(232, 232), dropout=args.posemb_dp, device=args.device)

        # Residual POS-Tagging Embedding
        self.res_posemb = EmbeddingLayer(embedding_size=(2 * args.lstm_hidden_size, 2 * args.lstm_hidden_size), dropout=None, device=args.device)

        # Bi-LSTM Encoder
        self.bilstm = DynamicLSTM(input_size=1000, hidden_size=args.lstm_hidden_size, num_layers=args.num_layers, dropout=args.bilstm_dp, bidirectional=True, device=args.device)

        # GCN
        self.gcns = nn.ModuleList()
        for i in range(args.gcn_num_layers):
            gcn = GraphConvolution(in_features=2 * args.lstm_hidden_size, out_features=2 * args.lstm_hidden_size, edge_types=args.edge_types_num, dropout=args.gcn_dp if i != args.gcn_num_layers - 1 else 0, use_bn=args.gcn_use_bn, device=args.device)
            self.gcns.append(gcn)


        # Highway
        if args.highway_use:
            self.hws = nn.ModuleList()
            for i in range(args.gcn_num_layers):
                hw = HighWay(size=2 * args.lstm_hidden_size, dropout_ratio=args.gcn_dp)
                self.hws.append(hw)


        self.sa_output = BottledXavierLinear(in_features=4 * args.lstm_hidden_size, out_features=args.sa_classes).to(device=args.device)

        # CRF
        self.CRF_model = CRF(4, batch_first=True)

        if args.target_method == 'BIO':
            self.dt_output = nn.Linear(4 * args.lstm_hidden_size, 4)
        else:
            self.dt_output = nn.Linear(4 * args.lstm_hidden_size, 3)

        self.loss_func_sa = FocalLoss(alpha=0.5, num_classes=4)

        self.dropout_sa = nn.Dropout(0.5) # 0.5
        self.dropout_dt = nn.Dropout(0.35) # 0.2 0.35

    def forward(self, sentence_cs, pos_class, text_len, adjmv, yes_no, target_zo, input_mask, segment_type_ids):
        mask = np.zeros(shape=sentence_cs.size(), dtype=np.uint8)
        for i in range(sentence_cs.size()[0]):
            s_len = int(text_len[i])
            mask[i, 0:s_len] = np.ones(shape=(s_len), dtype=np.uint8)
        mask = torch.ByteTensor(mask).to(self.args.device)[:,1:]

        word_emb, _ = self.wembeddings(sentence_cs, token_type_ids=segment_type_ids, attention_mask=input_mask)
        pos_emb = self.pembeddings(pos_class)
        x_emb = torch.cat([word_emb, pos_emb], 2)  # (batch_size, seq_len, d)

        cs_emb = x_emb[:, 1:3, :]
        text_emb = x_emb[:, 4:torch.max(text_len)+2, :]
        x = torch.cat([cs_emb, text_emb], dim=1)

        x, _ = self.bilstm(x, text_len)  # (batch_size, seq_len, d')

        # gcns
        for i in range(self.args.gcn_num_layers):
            if self.args.highway_use:
                x = self.gcns[i](x, adjmv) # + self.hws[i](x)  # (batch_size, seq_len, d')
            else:
                x = self.gcns[i](x, adjmv)

        # # gacns
        # for i in range(self.args.gcn_num_layers):
        #     if self.args.highway_use:
        #         x = self.gacns[i](x, adjmv) + self.hws[i](x)  # (batch_size, seq_len, d')
        #     else:
        #         x = self.gacns[i](x, adjmv)

        logits_sa = self.sa_output(self.dropout_sa(x[:, 0:2, :].view([x.size()[0], -1])))
        loss_sa = self.loss_func_sa(logits_sa, yes_no)

        res_pos = pos_class[:, 4:torch.max(text_len)+2]
        res_pos_emb = self.res_posemb(res_pos)
        merge_pos = self.dropout_dt(torch.cat([x[:, 2:torch.max(text_len)+2, :], res_pos_emb], dim=-1))
        logits_dt = self.dt_output(merge_pos)

        loss_fct_dt = CrossEntropyLoss(ignore_index=2) # weight=weight_dt,
        loss_dt = loss_fct_dt(logits_dt.view(-1, 3), target_zo.view(-1))

        return loss_sa, loss_dt, logits_sa, logits_dt, mask
