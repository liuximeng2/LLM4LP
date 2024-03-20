import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from transformers import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from peft import get_peft_model, LoraConfig
from torch_sparse import SparseTensor
import os

def make_imatrix(num_nodes, device = 'cuda'):
    edge_list = [(i, i) for i in range(num_nodes)]
    edge_list = torch.tensor(edge_list).to(device)
    edge_list = torch.cat((edge_list.t(), edge_list.t()[[1,0]]),dim=1)
    edge_weight_mask = torch.ones(edge_list.size(1)).to(torch.float).to(device)
    adj = SparseTensor.from_edge_index(edge_list, edge_weight_mask, [num_nodes, num_nodes]).to(device)
    return adj

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, mlp_layer=None, head=None, node_num=None,  cat_node_feat_mf=False, data_name=None):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()

        if data_name == 'ogbl-citation2':
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels,normalize=False ))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels, normalize=False))
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        
        else:
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels))
                self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):

        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class mlp_score(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout = 0.0):
        super(mlp_score, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 1, dropout = 0.0):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
    
    def forward(self, x):

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x


class BertClassifier(nn.Module):
    def __init__(self, model, score_func, batch_size, dropout = 0.0):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout)
        self.score_func = score_func
        #self.gnn = GCN(512, 128, 128, 1, 0.3)
        #self.batch_size = batch_size

    def forward(self, input_ids, attention_mask, batch_size, token_max_length = 512, pseudo_pos = None, pseudo_neg = None):
        
        input_ids = torch.reshape(input_ids, (-1, token_max_length))
        attention_mask = torch.reshape(attention_mask, (-1, token_max_length))
        #input_ids = input_ids.resize(batch_size * 4, token_max_length)
        #attention_mask = attention_mask.resize(batch_size * 4, token_max_length)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)

        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        emb = sum_embeddings / sum_mask

        pos_output = self.score_func(emb[0::2][::2], emb[1::2][::2])
        pos_loss = -torch.log(pos_output + 1e-15).mean()

        neg_out = self.score_func(emb[0::2][1::2], emb[1::2][1::2])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss_labeled = pos_loss + neg_loss

        return pos_output, neg_out, loss_labeled
    
    def initialize(self, num_tuned_layers):
        for param in self.bert.parameters():
            param.requires_grad = False
        if num_tuned_layers >= 1:
            for i in range(1, num_tuned_layers + 1):
                for param in self.bert.encoder.layer[i * -1].parameters():
                    param.requires_grad = True
    
    def reset_parameters(self, pretrained_model):
        self.bert = pretrained_model
        self.score_func.reset_parameters()
        #self.gnn.reset_parameters()

class BertClassifierInf(nn.Module):
    def __init__(self, model):
        super(BertClassifierInf, self).__init__()
        self.bert = model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)
        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        emb = sum_embeddings / sum_mask
        #emb = F.normalize(emb, p=2, dim=1)
        return emb
    

class BertClassifierReg(nn.Module):
    def __init__(self, model, score_func, batch_size, dropout = 0.0):
        super(BertClassifierReg, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout)
        self.score_func = score_func

    def forward(self, input_ids, attention_mask, pseudo_input_ids, pseudo_attention_mask, pseudo_score, token_max_length = 512):
        
        input_ids = torch.reshape(input_ids, (-1, token_max_length)).to('cuda')
        attention_mask = torch.reshape(attention_mask, (-1, token_max_length)).to('cuda')
        pseudo_input_ids = torch.reshape(pseudo_input_ids, (-1, token_max_length)).to('cuda')
        pseudo_attention_mask = torch.reshape(pseudo_attention_mask, (-1, token_max_length)).to('cuda')

        outputs = self.bert(input_ids=torch.cat((input_ids, pseudo_input_ids), 0), attention_mask=torch.cat((attention_mask, pseudo_attention_mask), 0), output_hidden_states = True)

        token_embeddings = outputs[0]
        input_mask_expanded = torch.cat((attention_mask, pseudo_attention_mask), 0).unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        emb = sum_embeddings / sum_mask

        pseudo_emb = emb[input_ids.size(0):, :]
        emb = emb[:input_ids.size(0), :]

        pos_output = self.score_func(emb[0::2][::2], emb[1::2][::2])
        pos_loss = -torch.log(pos_output + 1e-15).mean()

        neg_out = self.score_func(emb[0::2][1::2], emb[1::2][1::2])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss_labeled = pos_loss + neg_loss

        output = self.score_func(pseudo_emb[::2], pseudo_emb[1::2])
        output = torch.reshape(output, (-1, ))
        mle = torch.nn.MSELoss()

        loss_pseudo = mle(output, pseudo_score.to('cuda')) * 5

        return pos_output, neg_out, loss_labeled, loss_pseudo
    
    def initialize(self, num_tuned_layers):
        for param in self.bert.parameters():
            param.requires_grad = False
        if num_tuned_layers >= 1:
            for i in range(1, num_tuned_layers + 1):
                for param in self.bert.encoder.layer[i * -1].parameters():
                    param.requires_grad = True
    
    def reset_parameters(self, pretrained_model):
        self.bert = pretrained_model
        self.score_func.reset_parameters()

class BertLoRAClassifier(nn.Module):
    def __init__(self, model, score_func, batch_size, dropout = 0.0):
        super(BertLoRAClassifier, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout)
        self.score_func = score_func

    def forward(self, input_ids, attention_mask, pseudo_input_ids, pseudo_attention_mask, pseudo_score, token_max_length = 512):
        
        input_ids = torch.reshape(input_ids, (-1, token_max_length))
        attention_mask = torch.reshape(attention_mask, (-1, token_max_length))
        pseudo_input_ids = torch.reshape(pseudo_input_ids, (-1, token_max_length))
        pseudo_attention_mask = torch.reshape(pseudo_attention_mask, (-1, token_max_length))

        outputs = self.bert(input_ids=torch.cat((input_ids, pseudo_input_ids), 0), attention_mask=torch.cat((attention_mask, pseudo_attention_mask), 0), output_hidden_states = True)

        token_embeddings = outputs[0]
        input_mask_expanded = torch.cat((attention_mask, pseudo_attention_mask), 0).unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        emb = sum_embeddings / sum_mask

        pseudo_emb = emb[input_ids.size(0):, :]
        emb = emb[:input_ids.size(0), :]

        pos_output = self.score_func(emb[0::2][::2], emb[1::2][::2])
        pos_loss = -torch.log(pos_output + 1e-15).mean()

        neg_out = self.score_func(emb[0::2][1::2], emb[1::2][1::2])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss_labeled = pos_loss + neg_loss

        output = self.score_func(pseudo_emb[::2], pseudo_emb[1::2])
        output = torch.reshape(output, (-1, ))
        mle = torch.nn.MSELoss()

        loss_pseudo = mle(output, pseudo_score) * 5

        return pos_output, neg_out, loss_labeled, loss_pseudo
    
    def initialize(self, num_tuned_layers):
        for param in self.bert.parameters():
            param.requires_grad = False
        if num_tuned_layers >= 1:
            for i in range(1, num_tuned_layers + 1):
                for param in self.bert.encoder.layer[i * -1].parameters():
                    param.requires_grad = True
    
    def reset_parameters(self, pretrained_model):
        self.bert = get_peft_model(pretrained_model, self.loraConfig)
        self.score_func.reset_parameters()

class BertLoRAClassifierInf(nn.Module):
    def __init__(self, model, score_func, batch_size, dropout = 0.0):
        super(BertLoRAClassifierInf, self).__init__()
        self.bert = get_peft_model(model, LoraConfig(inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.1))
        self.dropout = nn.Dropout(dropout)
        self.score_func = score_func

    def forward(self, input_ids, attention_mask, batch_size, token_max_length = 512):
        
        input_ids = torch.reshape(input_ids, (-1, token_max_length))
        attention_mask = torch.reshape(attention_mask, (-1, token_max_length))
        #input_ids = input_ids.resize(batch_size * 4, token_max_length)
        #attention_mask = attention_mask.resize(batch_size * 4, token_max_length)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)

        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        emb = sum_embeddings / sum_mask

        pos_output = self.score_func(emb[0::2][::2], emb[1::2][::2])
        pos_loss = -torch.log(pos_output + 1e-15).mean()

        neg_out = self.score_func(emb[0::2][1::2], emb[1::2][1::2])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss_labeled = pos_loss + neg_loss

        return pos_output, neg_out, loss_labeled