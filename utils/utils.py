import os
import numpy as np
import torch
from torch_sparse import SparseTensor
from torch_geometric.transforms import RandomLinkSplit
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto import AutoTokenizer
import random
import logging, sys
import math
import logging.config 

def find_tensor_names(tensor_id):
    for name, obj in globals().items():
        if id(obj) == tensor_id:
            return name
    return "Not found"

def make_imatrix(num_nodes, device = 'cuda'):
    edge_list = [(i, i) for i in range(num_nodes)]
    edge_list = torch.tensor(edge_list).to(device)
    edge_list = torch.cat((edge_list.t(), edge_list.t()[[1,0]]),dim=1)
    edge_weight_mask = torch.ones(edge_list.size(1)).to(torch.float).to(device)
    adj = SparseTensor.from_edge_index(edge_list, edge_weight_mask, [num_nodes, num_nodes]).to(device)
    return adj

def parameter_counter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_data(dataset, seed = 2, num_val = 0.05, num_test = 0.1):
    init_seed(seed)
    data = torch.load(f'data/{dataset}/{dataset}_fixed_sbert.pt', map_location='cpu')
    num_nodes = int(data.y.size()[0])
    data.num_nodes = num_nodes
    edge_weight = torch.ones(data.edge_index.size(1))
    adj = SparseTensor.from_edge_index(data.edge_index, edge_weight, [num_nodes, num_nodes])
    data.adj = adj

    data.edge_index = data.edge_index[:,::2]#edge index [2, num_edge]
    transform = RandomLinkSplit(is_undirected=False, add_negative_train_samples= True, num_val = num_val, num_test = num_test, split_labels=True)
    train_data, val_data, test_data = transform(data)

    edge_to_index = {tuple(edge): idx for idx, edge in enumerate(data.edge_index.t().tolist())}
    train_indice = [edge_to_index[tuple(edge)] for edge in train_data.pos_edge_label_index.t().tolist()]
    gold_label = torch.zeros(data.edge_index.size()[1], dtype = torch.bool)
    gold_label[train_indice] = True
    data.gold_label = gold_label

    data.train_edges =  train_data.pos_edge_label_index.transpose(1,0) #[num_edge, 2]
    data.train_edges_false = train_data.neg_edge_label_index.transpose(1,0)
    data.val_edges = val_data.pos_edge_label_index.transpose(1,0)
    data.val_edges_false = val_data.neg_edge_label_index.transpose(1,0)
    data.test_edges = test_data.pos_edge_label_index.transpose(1,0)
    data.test_edges_false = test_data.neg_edge_label_index.transpose(1,0)

    idx = torch.randperm(torch.Tensor(data.train_edges).size(0))
    idx = idx[:torch.Tensor(data.val_edges).size(0)]
    data.train_val = torch.Tensor(data.train_edges)[idx]

    bertSmall_x = torch.load(f'data/{dataset}/{dataset}_bert-small.pt')
    data.bertSmall_x = bertSmall_x
    
    return data

def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'

class textDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].size()[0]
    
    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        return item

def save_embedding(data, model, save_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/bert/")
    X = tokenizer(data.raw_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    text_dataset = textDataset(X)
    text_dataloader = DataLoader(text_dataset, batch_size= batch_size, shuffle=False)

    embeddings = []
    for batch in text_dataloader:
        with torch.no_grad():  # Inference mode, no gradients needed
            batch_embeddings = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))  # Your model generates embeddings
            embeddings.append(batch_embeddings)

    # Concatenate all embeddings (if they are torch tensors)
    all_embeddings = torch.cat(embeddings, dim=0)
    torch.save(all_embeddings, save_path)


def get_root_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "")


def get_config_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "config")


def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
        
def save_model(model, save_path, emb=None):

    if emb == None:
        state = {
            'state_dict_model'	: model.state_dict(),
            # 'state_dict_predictor'	: linkPredictor.state_dict(),
        }

    else:
        state = {
            'state_dict_model'	: model.state_dict(),
            'emb'	: emb.weight
        }

    torch.save(state, save_path)

def save_emb(score_emb, save_path):

    if len(score_emb) == 6:
        pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, x1= score_emb
        state = {
        'pos_train_pred': pos_train_pred,
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x1
        }
        
    elif len(score_emb) == 5:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x
        }
    elif len(score_emb) == 4:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, = score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        }
   
    torch.save(state, save_path)
    return state

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                values, indices = torch.topk(r[:, 1], 2)
                train2 = (r[indices[0], 0].item() +  r[indices[1], 0].item()) / 2#选择valid最大的result对应的train result
                test = (r[indices[0], 2].item() +  r[indices[1], 2].item()) / 2   #选择valid最大的result对应的test result
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')

            r = best_result[:, 0].float()
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1].float()
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 2].float()
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 3].float()
            print(r)
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]


            return best_valid, best_valid_mean, mean_list, var_list


class Logger_ddi(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.epoch_num = 10

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, eval_step, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            # argmax = result[:, 1].argmax().item()
            for i in range(result.size(0)):
                if (i+1)%self.epoch_num == 0:

                    print(f'Run {run + 1:02d}:')
                    print(f'Epoch {(i + 1)*eval_step:02d}:')
                    print(f'Train: {result[i, 0]:.2f}')
                    print(f'Valid: {result[i, 1]:.2f}')
                    print(f'Test: {result[i, 2]:.2f}')
        else:
            # result = 100 * torch.tensor(self.results)

            # best_results = []
            
            eval_num = int(len(self.results[0])/self.epoch_num)
            all_results = [[] for _ in range(eval_num)]

            for r in self.results:
                r = 100 * torch.tensor(r)

                for i in range(r.size(0)):
                    if (i+1)%self.epoch_num == 0:

                        train = r[i, 0].item()
                        valid = r[i, 1].item()
                        test = r[i, 2].item()
                
                        all_results[int((i+1)/self.epoch_num)-1].append((train, valid, test))


            for i, best_result in enumerate(all_results):
                best_result = torch.tensor(best_result)


                print(f'All runs:')
                
                epo = (i + 1)*self.epoch_num
                epo = epo*eval_step
                print(f'Epoch {epo:02d}:')


                # r = best_result[:, 0]
                # print(f'Final Train: {r.mean():.2f} ± {r.std():.2f}')

                r = best_result[:, 0]
                best_train_mean = round(r.mean().item(), 2)
                best_train_var = round(r.std().item(), 2)
                print(f'Final Train: {r.mean():.2f} ± {r.std():.2f}')

                r = best_result[:, 1]
                best_valid_mean = round(r.mean().item(), 2)
                best_valid_var = round(r.std().item(), 2)

                best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
                print(f'Final Valid: {r.mean():.2f} ± {r.std():.2f}')


                r = best_result[:, 2]
                best_test_mean = round(r.mean().item(), 2)
                best_test_var = round(r.std().item(), 2)
                print(f'Final Test: {r.mean():.2f} ± {r.std():.2f}')

                mean_list = [best_train_mean, best_valid_mean, best_test_mean]
                var_list = [best_train_var, best_valid_var, best_test_var]


            # return best_valid, best_valid_mean, mean_list, var_list


def get_logger(name, log_dir, config_dir):
	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def get_memory():
    t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    r = torch.cuda.memory_reserved(0) / 1024 / 1024
    a = torch.cuda.memory_allocated(0) / 1024 / 1024
    return f"Allocated memory: {(r + a):.2f}MB / Total memory: {t:.2f}MB"

def find_first_zero_indices(matrix):
    # Convert to a NumPy array if not already
    np_matrix = np.array(matrix)
    # Initialize an array to hold the index of the first zero for each row
    indices = np.full(np_matrix.shape[0], -1) # Fill with -1 to indicate rows without a zero
    for i, row in enumerate(np_matrix):
        # Find the index of the first zero in the current row
        zero_index = np.where(row == 0)[0]
        if zero_index.size > 0:
            indices[i] = zero_index[0]
    return indices.min()