import os
from utils import load_rawtext_data
import torch
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig
from transformers.models.auto import AutoModel, AutoTokenizer
from model import mlp_score, MLP, MLPClassifier, MLPClassifierInf
from transformers.trainer import Trainer, TrainingArguments
from evaluator import evaluate_hits, evaluate_mrr, evaluate_auc
from torch.utils.data import DataLoader
import numpy as np
import logging
import warnings
from ogb.linkproppred import Evaluator

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(filename='Logs/MLP_train.log', level=logging.INFO)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, feature, edge, neg_edge, num_node):
        self.feature = feature
        self.edge = edge
        self.neg_edge = neg_edge
        self.num_node = num_node

    def __getitem__(self, idx):
        selected_pos_nodes = self.edge[idx].flatten().to(torch.long)
        selected_neg_nodes = self.neg_edge[idx].flatten().to(torch.long)
        pos_indice = torch.zeros(self.num_node).to(torch.bool)
        pos_indice[selected_pos_nodes] = True
        neg_indice = torch.zeros(self.num_node).to(torch.bool)
        neg_indice[selected_neg_nodes] = True
        feature = torch.cat((self.feature[pos_indice], self.feature[neg_indice]), dim = 0)

        item = {
            'x': feature,
        }
        return item
    
data, adj = load_rawtext_data('cora', 0)
num_node = data.x.size(0)
train_edge = data.train_edges
neg_train_edge = data.train_edges_false
num_edge = train_edge.size()[0]
feature = data.x
logging.info(f"# of train edge: {num_edge}")

text_dataset = TextDataset(feature, train_edge, neg_train_edge, num_node)

idx = np.arange(0, num_edge, 1)
train_dataset = torch.utils.data.Subset(text_dataset, idx)

batch_size = 32
hidden_layer = 256
num_layer = 4
mlp_dropout = 0.4
mlp = MLP(384, 256, 768, num_layers = 5, dropout = 0.3)
score_func = mlp_score(768, hidden_layer, 1 , 2 , mlp_dropout)
model = MLPClassifier(mlp, score_func).to(device)

num_epoch = 100
running_loss = 0
total_steps = len(train_dataset) * num_epoch
warmup_step = int(0.1 * total_steps)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = total_steps)

for epoch in range(num_epoch):
    model.train()
    for i, data in enumerate(DataLoader(train_dataset, batch_size, drop_last=True)):
        optimizer.zero_grad() 
        x = data['x'].to(device)
        loss = model(x)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if i % 10 == 9:
            logging.info(f'[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 10 :.3f}')
            running_loss = 0

torch.save(model.state_dict(), "ft_lm/cora_mlp.pt")
print('Model Saved!')

PATH = 'ft_lm/cora_mlp.pt'
model = MLPClassifierInf(mlp, score_func)
model.load_state_dict(torch.load(PATH))

data, _ = load_rawtext_data('cora', 0)
val_edge = data.val_edges
neg_val_edge = data.val_edges_false
test_edge = data.test_edges
neg_test_edge = data.test_edges_false

text_dataset = TextDataset(feature, val_edge, neg_val_edge, num_node)
val_dataset = torch.utils.data.Subset(text_dataset, np.arange(0, val_edge.size()[0]-1, 1))
text_dataset = TextDataset(feature, test_edge, neg_test_edge, num_node)
test_dataset = torch.utils.data.Subset(text_dataset, np.arange(0, test_edge.size()[0]-1, 1))

def test(dataset, batch_size, model):
    model.eval()
    pos_preds = []
    neg_preds = []
    for i, data in enumerate(DataLoader(dataset, batch_size, drop_last=True)):
            x = data['x'].to(device)
            pos_output, neg_output = model(x)
            pos_preds += pos_output
            neg_preds += neg_output
    pos_preds = torch.cat(pos_preds, dim = 0)
    neg_preds = torch.cat(neg_preds, dim = 0)
    return pos_preds, neg_preds

with torch.no_grad():
    logging.info('Training dataset')
    train_pos_preds, train_neg_preds = test(train_dataset, batch_size, model)
    logging.info('Validation dataset')
    val_pos_preds, val_neg_preds = test(val_dataset, batch_size, model)
    logging.info('Testing dataset')
    test_pos_preds, test_neg_preds = test(test_dataset, batch_size, model)

train_pos_preds = torch.flatten(train_pos_preds)
val_pos_preds = torch.flatten(val_pos_preds)
test_pos_preds = torch.flatten(test_pos_preds)

train_neg_preds = train_neg_preds.squeeze(-1)
val_neg_preds = val_neg_preds.squeeze(-1)
test_neg_preds = test_neg_preds.squeeze(-1)

result = {}
k_list = [1, 3, 10, 100]
evaluator_mrr = Evaluator(name='ogbl-citation2')
result_mrr_train = evaluate_mrr(evaluator_mrr, train_pos_preds, train_neg_preds)
result_mrr_val = evaluate_mrr(evaluator_mrr, val_pos_preds, val_neg_preds)
result_mrr_test = evaluate_mrr(evaluator_mrr, test_pos_preds, test_neg_preds)
result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])

for K in [1,3,10, 100]:
    result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])


for key, results in result.items():
                        
    logging.info(key)
    
    train_hits, valid_hits, test_hits = results
    logging.info(
                f'Train: {100 * train_hits:.2f}%, '
                f'Valid: {100 * valid_hits:.2f}%, '
                f'Test: {100 * test_hits:.2f}%')

    logging.info('---')