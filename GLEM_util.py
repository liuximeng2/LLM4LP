import torch

#no regression
def get_pesudo_edge(data, pred, threshold):
            
    pos_pesudo_edge = torch.cat((data.test_edges[pred['pos_test_score'] > threshold],
                        data.test_edges_false[pred['neg_test_score'] > threshold],
                        data.val_edges[pred['pos_valid_score'] > threshold],
                        data.val_edges_false[pred['neg_valid_score'] > threshold]), dim = 0)

    neg_pesudo_edge = torch.cat((data.test_edges[pred['pos_test_score'] < threshold],
                        data.test_edges_false[pred['neg_test_score'] < threshold],
                        data.val_edges[pred['pos_valid_score'] < threshold],
                        data.val_edges_false[pred['neg_valid_score'] < threshold]), dim = 0)
    
    return pos_pesudo_edge, neg_pesudo_edge

#regression
def get_pseudo_score(data, pred):

    pesudo_edge = torch.cat((data.test_edges,
                             data.test_edges_false,
                             data.val_edges,
                             data.val_edges_false), dim = 0)
    
    pseudo_score = torch.cat((pred['pos_test_score'],
                              pred['neg_test_score'],
                              pred['pos_valid_score'],
                              pred['neg_valid_score']), dim = 0)
    
    return pesudo_edge, pseudo_score

def get_indice(edge, neg_edge, idx, num_node):
    
    selected_pos_nodes = edge[idx].flatten().to(torch.long)
    if neg_edge != None:
        selected_neg_nodes = neg_edge[idx].flatten().to(torch.long)
    else:
        selected_neg_nodes = torch.randperm(num_node)[:2].to(torch.long).to(selected_pos_nodes.device)
    pos_indice = torch.zeros(num_node).to(torch.bool)
    pos_indice[selected_pos_nodes] = True
    neg_indice = torch.zeros(num_node).to(torch.bool)
    neg_indice[selected_neg_nodes] = True

    return pos_indice, neg_indice