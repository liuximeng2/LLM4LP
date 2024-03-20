import os
from utils import *
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.auto import AutoModel, AutoTokenizer
from model import BertClassifier, BertClassifierInf
from transformers.trainer import Trainer, TrainingArguments
from model import mlp_score
from evaluator import evaluate_hits, evaluate_mrr, evaluate_auc
from torch.utils.data import DataLoader
import numpy as np
import logging
import warnings
from ogb.linkproppred import Evaluator
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import AutoModel, GPT2Model
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType, LoftQConfig

data = torch.load(f'data/cora/cora_fixed_sbert.pt', map_location='cpu')
cora = load_data('cora')
print(cora.raw_text[0])
#print(citeseer.val_edges.size())

#print(any((pubmed.train_edges == row).all(1).any() for row in pubmed.val_edges))
#print(cora_entity[0])
#print(cora_explanation[0])
'''
pretrained_model = AutoModel.from_pretrained("model/pre_train/bert-small/")
score_func = mlp_score(128, 128, 1 , 3 , 0.3)
model = BertClassifier(pretrained_model, score_func, 64)
model.initialize(0)
print(parameter_counter(model.gnn))
print(parameter_counter(score_func))
'''