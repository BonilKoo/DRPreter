import argparse
import math
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_scatter import scatter_add
from torch_geometric.data import Batch

from drug_graph import *
from utils import *
from Model.DRPreter import DRPreter

def parse_args():
    path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', default=f'{path}/input.csv', help='Input csv file for prediction. First column: name | Second column: canonical_SMILES')
    parser.add_argument('--output', default=f'{path}/Result/output.csv', help='File name to save')
    parser.add_argument('--top_k', default=10, help='The number of genes with high importance scores to store in the file')
    
    parser.add_argument('--device', type=int, default=0, help='device')
    
    parser.add_argument('--cell_info', default=f'{path}/Data/Cell/cell_line_info.csv')
    parser.add_argument('--cell_dict', default=f'{path}/Data/Cell/cell_feature_std.pkl', help='Cell graph')
    parser.add_argument('--edge_index', default=f'{path}/Data/Cell/edge_index.npy', help='STRING edges')
    parser.add_argument('--model', default=f'{path}/weights/weight_seed245.pth', help='Trained DRPreter model')
    parser.add_argument('--gene_dict', default=f'{path}/Data/Cell/cell_idx2gene_dict.pkl', help='A dictionary to map indices to gene names')
    parser.add_argument('--pathway_dict', default=f'{path}/Data/Cell/34pathway_score990.pkl', help='A dictionary of pathways and genes belonging to them')
    
    args, unknown = parser.parse_known_args()
    return args

def drug_to_graph(input_data):
    drug_dict = {}
    for idx in range(len(input_data)):
        drug_dict[input_data.loc[idx, input_data.columns[0]]] = smiles2graph(input_data.loc[idx, input_data.columns[1]])
    return drug_dict

def predict(model, drug_dict, cell_dict, edge_index, args, data):
    model.eval()
    
    IC50_pred = []
    with torch.no_grad():
        for cell in data[data.columns[2]].unique():
            cell_dict[cell].edge_index = torch.tensor(edge_index, dtype=torch.long)
        drug_list = [drug_dict[name] for name in data[data.columns[0]]]
        cell_list = [cell_dict[name] for name in data[data.columns[2]]]
        batch_size = 2048
        batch_num = math.ceil(len(drug_list) / batch_size)
        for index in tqdm(range(batch_num)):
            drug = Batch.from_data_list(drug_list[index*batch_size:(index+1)*batch_size]).to(args.device)
            cell = Batch.from_data_list(cell_list[index*batch_size:(index+1)*batch_size]).to(args.device)
            y_pred = model(drug, cell)
            IC50_pred.append(y_pred)
        IC50_pred = torch.cat(IC50_pred, dim=0)
    data['lnIC50'] = IC50_pred.detach().cpu().numpy()
    data['IC50'] = np.exp(data['lnIC50'])
    torch.cuda.empty_cache()
    
    return data

def gene_importance_score(data, model, drug_dict, cell_dict, edge_index, args):
    with open(args.gene_dict, 'rb') as file:
        gene_dict = pickle.load(file)
    
    total_gene_df = pd.Series(list(range(len(data))))
    for index in tqdm(range(len(data))):
        drug_name, cell_name = data.iloc[index, [0,2]]
        _, indices = gradcam(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
        idx2gene = [gene_dict[idx] for idx in indices]
        gene_df = pd.DataFrame(idx2gene)
        total_gene_df.loc[index] = '|'.join(list(gene_df.drop_duplicates(keep='first')[0])[:args.top_k])
    
    data[f'GeneSymbolList'] = total_gene_df
    
    return data

def pathway_attention_score(data, model, drug_dict, cell_dict, edge_index, args):
    with open(args.pathway_dict, 'rb') as file:
        pathway_names = pickle.load(file)
    tks = [p[5:] for p in list(pathway_names)]
    
    total_pathway_df = pd.Series(list(range(len(data))))
    for index in tqdm(range(len(data))):
        drug_name, cell_name = data.iloc[index, [0,2]]
        attn_score = attention_score(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
        attn_score = torch.squeeze(attn_score).mean(axis=0).detach().cpu().numpy()[:-1, :-1]
        average_attn_score = attn_score.sum(axis=0) / args.n_pathways
        average_attn_score = pd.DataFrame(average_attn_score, index=tks, columns=['score']).sort_values('score', ascending=False)
        if len(np.unique(average_attn_score['score'])) == 1:
            total_pathway_df.loc[index] = ''
        else:
            average_attn_score = average_attn_score[average_attn_score['score'] != 0]
            total_pathway_df.loc[index] = '|'.join(average_attn_score.index.to_list()[:args.top_k])
    
    data[f'KEGGPathwayList'] = total_pathway_df
    if '/' in args.output:
        os.makedirs('/'.join(args.output.split('/')[:-1]), exist_ok=True)
    data.to_csv(args.output, index=False)
    
    return data

def main(args):
    
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    args.batch_size = 1
    args.seed = 42
    args.layer = 3
    args.hidden_dim = 8
    args.layer_drug = 3
    args.dim_drug = 128
    args.dim_drug_cell = 256
    args.dropout_ratio = 0.1
    args.trans = True
    
    data = pd.read_csv(args.input)
    data = data[['DrugName', 'Canonical_SMILES']]
    cell_info = pd.read_csv(args.cell_info)
    with open(args.cell_dict, 'rb') as file:
        cell_dict = pickle.load(file) # pyg data format of cell graph
    edge_index = np.load(args.edge_index)
    data = data.merge(cell_info, how='cross')
    data['lnIC50'] = np.nan
    
    drug_dict = drug_to_graph(data)    
    
    for key in cell_dict.keys():
        example_cell = cell_dict[key]
        args.num_genes, args.num_cell_feature = example_cell.x.shape
        break
    for key in drug_dict.keys():
        example_drug = drug_dict[key]
        args.num_drug_feature = example_drug.x.shape[1]
    
    gene_list = scatter_add(torch.ones_like(example_cell.x.squeeze()), example_cell.x_mask.to(torch.int64)).to(torch.int)
    args.n_pathways = gene_list.size(0)
    args.max_gene = gene_list.max().item()
    args.cum_num_nodes = torch.cat([gene_list.new_zeros(1), gene_list.cumsum(dim=0)], dim=0)
    
    _, _, test_loader = load_data(data, drug_dict, cell_dict, torch.tensor(edge_index, dtype=torch.long), args, val_ratio=0, test_ratio=1)
    
    model = DRPreter(args).to(args.device)
    
    model.load_state_dict(torch.load(args.model, map_location=args.device)['model_state_dict'])
    
    data = predict(model, drug_dict, cell_dict, edge_index, args, data)
    data = gene_importance_score(data, model, drug_dict, cell_dict, edge_index, args)
    data = pathway_attention_score(data, model, drug_dict, cell_dict, edge_index, args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
