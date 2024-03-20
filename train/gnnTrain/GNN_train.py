import argparse
from utils.utils import *
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from ogb.linkproppred import Evaluator
from utils.utils import evaluate_hits, evaluate_mrr, evaluate_auc

dir_path  = get_root_dir()

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 3, 10, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [1, 3, 10, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])


    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    # for K in [1,3,10, 100]:
    #     result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

   
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])

    
    return result

def train(model, score_func, train_pos,train_neg, x, optimizer, batch_size, args, pesudo_pos = None, pesudo_neg = None):
    model.train()
    score_func.train()
    total_loss = total_examples = 0
    '''
    if pesudo_pos == None:
        len = len(DataLoader(range(train_pos.size(0)), batch_size, shuffle=True))
        len = int(max(pesudo_pos.size()[0], pesudo_neg.size()[0]) / len)
        print(len)
    '''
        
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):

        optimizer.zero_grad()
        num_nodes = x.size(0)

        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1, 0)

        if args.structure == True:
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
            edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
            adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        
        else:
            adj = make_imatrix(num_nodes, train_pos.device)
        ###################

        h = model(x, adj)

        edge = train_pos[perm].t().to(torch.long)

        if args.self_train == True and pesudo_pos != None:
            pesudo_perm = torch.randperm(min(pesudo_pos.size()[0], pesudo_neg.size()[0]))
            pesudo_pos_shuff = pesudo_pos[pesudo_perm[:256], :].to(edge.device)
            edge = torch.cat((edge, pesudo_pos_shuff.t().to(torch.long)), dim = 1)

        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.

        #edge = train_neg[perm].t().to(torch.long)

        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long, device=h.device)
        
        if args.self_train == True and pesudo_neg != None:
            pesudo_neg_shuff = pesudo_neg[pesudo_perm[:256], :].to(edge.device)
            edge = torch.cat((edge, pesudo_neg_shuff.t().to(torch.long)), dim = 1)

        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size):

    # input_data  = input_data.transpose(1, 0)
    # with torch.no_grad():
    preds = []
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t().to(torch.long)
    
        preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
        
    pred_all = torch.cat(preds, dim=0)

    return pred_all


@torch.no_grad()
def test(model, score_func, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    
    
    h = model(x, data.adj.to(x.device))
    # print(h[0][:10])
    x = h

    pos_train_pred = test_edge(score_func, data.train_val, h, batch_size)

    neg_valid_pred = test_edge(score_func, data.val_edges_false, h, batch_size)

    pos_valid_pred = test_edge(score_func, data.val_edges, h, batch_size)

    pos_test_pred = test_edge(score_func, data.test_edges, h, batch_size)

    neg_test_pred = test_edge(score_func, data.test_edges_false, h, batch_size)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)


    #logging.info('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_train_pred.cpu(), pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]

    return result, score_emb




def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='arxiv')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1) # number of GCN layers
    parser.add_argument('--num_layers_predictor', type=int, default=3) # number of decoder layers
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default = 1)
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=5, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2', type=float, default=0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ###### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')

    ###### mode
    parser.add_argument('--mode', type = str, default='lm_embedding')
    parser.add_argument('--structure', default=True, help='including graph strucutre')
    parser.add_argument('--lm_model', type=str, default='bert-small')
    parser.add_argument('--edge_split_seed', type=int, default=2)
    parser.add_argument('--self_train', type=bool, default=False)
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--regression',  type = bool, default=True)    
    
    args = parser.parse_args()

    log_path = f'Logs/LM_GNN_train/{args.data_name}/{args.lm_model}/{args.structure}/es({args.edge_split_seed})/st_lr({args.lr})_mlp_layer({args.num_layers_predictor})_drop({args.dropout})_l2({args.l2}).log'
    init_path(log_path)
    logging.basicConfig(filename= log_path, level=logging.INFO)
    logging.info(f'Edge split seed: {args.edge_split_seed}')
    logging.info(args)

    data = load_data(args.data_name, args.edge_split_seed)
    logging.info(data)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.lm_model == 'bert-small':
        x = data.bertSmall_x
    if args.lm_model == 'sbert':
        x = data.x

    logging.info(f'{data.train_edges.size()}, {data.val_edges.size()}, {data.test_edges.size()}')

    node_num = data.x.size(0)
    x = x.to(device)

    if args.structure == False:
        data.adj = make_imatrix(node_num, device)

    train_pos = data.train_edges.to(x.device)
    train_neg = data.train_edges_false.to(x.device)

    input_channel = x.size(1)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels, 
                                 args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num, args.cat_node_feat_mf).to(device)
    
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels, 
                                        1, args.num_layers_predictor, args.dropout).to(device)
    logging.info(f'# of parameters: {parameter_counter(model) + parameter_counter(score_func)}')
    
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    pos_pesudo_edge, neg_pesudo_edge = [], []

    loggers = {
        'MRR': Logger(args.runs),
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs)
    }

    for run in range(args.runs):

        logging.info(f'################################# run {run} #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        logging.info(f'seed: {seed}')

        init_seed(seed)
        score_save_path = f'GLEM_model/GNN/{args.gnn_model}/embedding/{args.data_name}/reg({args.regression})/iter(0)/es({args.edge_split_seed})/seed({args.seed})/lr({args.lr})_do_({args.dropout})_seed({args.seed})'
        model_save_path = f'GLEM_model/GNN/{args.gnn_model}/model/{args.data_name}/reg({args.regression})/iter(0)/es({args.edge_split_seed})/seed({args.seed})/gnn_lr({args.lr})_do_({args.dropout})_seed({args.seed})'
        score_func_save_path = f'GLEM_model/GNN/{args.gnn_model}/model/{args.data_name}/reg({args.regression})/iter(0)/es({args.edge_split_seed})/seed({args.seed})/sf_lr({args.lr})_do_({args.dropout})_seed({args.seed})'
        init_path(model_save_path)
        init_path(score_save_path)

        model.reset_parameters()
        score_func.reset_parameters()

        optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):

            if args.self_train == True and epoch > args.eval_steps:
                loss = train(model, score_func, train_pos,train_neg, x, optimizer, args.batch_size, args, pos_pesudo_edge, neg_pesudo_edge)


            loss = train(model, score_func, train_pos,train_neg, x, optimizer, args.batch_size, args)
            # print(model.convs[0].att_src[0][0][:10])
            
            if epoch % args.eval_steps == 0:
                results_rank, score_emb = test(model, score_func, data, x, evaluator_hit, evaluator_mrr, args.batch_size)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        if key == 'MRR':
                        
                            #logging.info(key)
                            
                            train_hits, valid_hits, test_hits = result


                            logging.info(
                                f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                    #logging.info('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0

                    if args.save:
                        save_emb(score_emb, score_save_path) 
                        torch.save(model.state_dict(), model_save_path)
                        torch.save(score_func.state_dict(), score_func_save_path)        
                    if args.self_train == True:
                        pos_pesudo_edge, neg_pesudo_edge = get_pesudo_edge(data, score_emb, 0.5)

                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        logging.info("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():

        best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()

        #logging.info(key)
        #logging.info(loggers[key].print_statistics())
        logging.info(f'{key} result: Train: {mean_list[0]} ± {var_list[0]}, Valid: {mean_list[1]} ± {var_list[1]}, Test: {mean_list[2]} ± {var_list[2]}')

        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean
            
        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]

    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()

   