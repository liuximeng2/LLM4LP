import os
from utils import *
import torch
from transformers import get_linear_schedule_with_warmup, AutoConfig
from transformers.models.auto import AutoModel, AutoTokenizer
from model import BertLoRAClassifier, BertLoRAClassifierInf, mlp_score
from transformers.trainer import Trainer, TrainingArguments
from evaluator import evaluate_hits, evaluate_mrr, evaluate_auc
from torch.utils.data import DataLoader
import numpy as np
from ogb.linkproppred import Evaluator
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, edge, num_node, neg_edge = None):
        self.encodings = encodings
        self.edge = edge
        self.neg_edge = neg_edge
        self.num_node = num_node
        self.num_edge = edge.size()[0]

    def __getitem__(self, idx):
        selected_pos_nodes = self.edge[idx].flatten().to(torch.long)
        if self.neg_edge != None:
            selected_neg_nodes = self.neg_edge[idx].flatten().to(torch.long)
        else:
            selected_neg_nodes = torch.randperm(num_node)[:2].to(torch.long).to(selected_pos_nodes.device)
        pos_indice = torch.zeros(self.num_node).to(torch.bool)
        pos_indice[selected_pos_nodes] = True
        neg_indice = torch.zeros(self.num_node).to(torch.bool)
        neg_indice[selected_neg_nodes] = True

        input_ids = torch.cat((self.encodings['input_ids'][pos_indice], self.encodings['input_ids'][neg_indice]), dim = 0)
        attention_mask = torch.cat((self.encodings['attention_mask'][pos_indice], self.encodings['attention_mask'][neg_indice]), dim = 0)

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        return item
    
    def __len__(self):
        return self.num_edge
    
def test(dataset, batch_size, model):
    model.eval()
    pos_preds = []
    neg_preds = []
    for i, data in enumerate(DataLoader(dataset, batch_size, drop_last=True, shuffle = True)):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            pos_output, neg_output, _ = model(input_ids, attention_mask, batch_size)
            pos_preds += pos_output
            neg_preds += neg_output
    pos_preds = torch.cat(pos_preds, dim = 0)
    neg_preds = torch.cat(neg_preds, dim = 0)
    return pos_preds, neg_preds

label = 1
model_name = 'bert-small-LoRA'
dataset = 'cora'

data = load_data(dataset, 2)   
num_node = data.x.size(0)
train_edge, neg_train_edge = data.train_edges, data.train_edges_false
val_edge, neg_val_edge = data.val_edges, data.val_edges_false
test_edge, neg_test_edge = data.test_edges, data.test_edges_false
num_train_edge = train_edge.size()[0]
num_test_edge = test_edge.size()[0]

logging.basicConfig(filename=f'Logs/LM_train/{dataset}/{model_name}_{label}.log', level=logging.INFO)

if model_name == 'bert-base':
    config = AutoConfig.from_pretrained('model/pre_train/bert/')
    config.attention_probs_dropout_prob = 0.1
    config.hidden_dropout_prob = 0.3
    pretrained_model = AutoModel.from_pretrained("model/pre_train/bert/", config = config)
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/bert/")
    X = tokenizer(data.raw_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    in_channel = 768
if model_name == 'bert-small-LoRA':
    config = AutoConfig.from_pretrained('model/pre_train/bert-small/')
    config.attention_probs_dropout_prob = 0.1
    config.hidden_dropout_prob = 0.3
    pretrained_model = AutoModel.from_pretrained("model/pre_train/bert-small/", config = config)
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/bert/")
    X = tokenizer(data.raw_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    in_channel = 512
if model_name == 'gpt2':
    config = AutoConfig.from_pretrained('model/pre_train/gpt2/')
    pretrained_model = AutoModel.from_pretrained("model/pre_train/gpt2/", config = config)
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/gpt2/")
    X = tokenizer(data.raw_texts, padding = True, truncation=True, max_length=512, return_tensors = 'pt')

#Creating training dataloaders
train_dataset = TextDataset(X, train_edge, num_node)
val_dataset = TextDataset(X, val_edge, num_node, neg_val_edge)
test_dataset = TextDataset(X, test_edge, num_node, neg_test_edge)
train_val_dataset = TextDataset(X, train_edge[:num_test_edge, :], num_node, neg_train_edge[:num_test_edge, :])

#Model parameters
num_tuned_layers = 4
hidden_layer = 256
num_layer = 2
mlp_dropout = 0.4

#Training parameters
runs = 1
batch_size = 32
num_epoch = 100
lr = 1e-4
l2 = 0
kill_cnt_step = 10
eval_step = 5

running_loss = 0
total_steps = (len(train_dataset) // batch_size) * num_epoch
warmup_step = int(0.1 * total_steps)

score_func = mlp_score(in_channel, hidden_layer, 1 , num_layer , mlp_dropout)
model = BertLoRAClassifier(pretrained_model, score_func).to(device)
print(parameter_counter(model))

logging.info(f"Model: {model_name}, Train/Val/Test: 85/5/10, seed: {label}, num_epoch: {num_epoch},")
logging.info(f"# lm_tuned_layer: {num_tuned_layers}, lm_attention_dropout: {config.attention_probs_dropout_prob}, lm_hidden_dropout: {config.hidden_dropout_prob}")
logging.info(f"scorefunc_dropout: {mlp_dropout}, scorefunc_hidden_layer: {hidden_layer}, scorefunc_num_layers: {num_layer}")
logging.info(f"batch_size: {batch_size}, learning_rate: {lr}, weight_decay: {l2}, kill_cnt: {kill_cnt_step}")
logging.info("--------------------------")

evaluator_hit = Evaluator(name='ogbl-collab')
evaluator_mrr = Evaluator(name='ogbl-citation2')

loggers = {
    'MRR': Logger(runs),
    'Hits@1': Logger(runs),
    'Hits@3': Logger(runs),
    'Hits@10': Logger(runs),
    'Hits@100': Logger(runs)
}

global_best_result = 0

for run in range(runs):

    early_stop = False
    best_result = 0
    kill_cnt = 0

    if runs == 1:
        seed = label
    else:
        seed = run
    init_seed(seed)
    pretrained_model = AutoModel.from_pretrained("model/pre_train/bert-small/", config = config).to(device)
    model.reset_parameters(pretrained_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay= l2)

    for epoch in range(1, 1 + num_epoch):
        for i, batch in enumerate(DataLoader(train_dataset, batch_size, drop_last=True, shuffle = True)):
            model.train()
            optimizer.zero_grad() 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _, _, loss = model(input_ids, attention_mask, batch_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            
        if (epoch + 1) % eval_step == 0:
            with torch.no_grad():
                train_pos_preds, train_neg_preds = test(train_val_dataset, batch_size, model)
                val_pos_preds, val_neg_preds = test(val_dataset, batch_size, model)
                test_pos_preds, test_neg_preds = test(test_dataset, batch_size, model)

            train_pos_preds = torch.flatten(train_pos_preds)
            val_pos_preds = torch.flatten(val_pos_preds)
            test_pos_preds = torch.flatten(test_pos_preds)
            train_neg_preds = torch.flatten(train_neg_preds)
            val_neg_preds = torch.flatten(val_neg_preds)
            test_neg_preds = torch.flatten(test_neg_preds)

            result = {}
            k_list = [1, 3, 10, 100]

            result_mrr_train = evaluate_mrr(evaluator_mrr, train_pos_preds, train_neg_preds.repeat(train_neg_preds.size(0), 1))
            result_mrr_val = evaluate_mrr(evaluator_mrr, val_pos_preds, val_neg_preds.repeat(val_neg_preds.size(0), 1))
            result_mrr_test = evaluate_mrr(evaluator_mrr, test_pos_preds, test_neg_preds.repeat(test_neg_preds.size(0), 1))
            result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])

            for K in [1,3,10, 100]:
                result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

            for key, results in result.items():

                loggers[key].add_result(run, results)
                            
                if key == 'MRR':
                    
                    train_hits, valid_hits, test_hits = results
                    logging.info(
                                f'Epoch: {epoch + 1}, '
                                f'train loss: {running_loss /(i * eval_step):.3f}, '
                                f'Train_mrr: {100 * train_hits:.2f}%, '
                                f'Valid_mrr: {100 * valid_hits:.2f}%, '
                                f'Test_mrr: {100 * test_hits:.2f}%')

                    if 100 * test_hits > best_result:
                        best_result = 100 * test_hits
                        kill_cnt = 0

                        if 100 * test_hits > global_best_result:
                            global_best_result = 100 * test_hits
                            #Saving model parameters
                            PATH =  f"ft_lm/model/{dataset}/{model_name}_{label}.pt"
                            torch.save(model.state_dict(), PATH)

                            #Saving node embeddings
                            emb_save_path = f'ft_lm/embedding/{dataset}/{model_name}_{label}.pt'
                            emb_model = BertLoRAClassifierInf(pretrained_model, score_func).to(device)
                            emb_model.load_state_dict(torch.load(PATH))
                            save_embedding(data, emb_model, emb_save_path, 32)
                            logging.info('Best Model Found, Embedding Saved')
                    else:
                        kill_cnt += 1

                        if kill_cnt > kill_cnt_step: 
                            logging.info("Early Stopping!!")
                            logging.info(f'Final test mrr: {100 * test_hits:.2f}, Best test mrr: {best_result:.2f}')
                            early_stop = True
                            break

            if early_stop == True:
                break

            running_loss = 0

#Show result
for key in loggers.keys():
    logging.info(key)
    logging.info(loggers[key].print_statistics())
