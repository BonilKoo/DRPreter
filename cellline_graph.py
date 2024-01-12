import numpy as np
import pandas as pd
import os
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import sparse
import pickle
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_data', default='./Data/Cell')
    parser.add_argument('--kegg', default='./Data/Cell/34pathway_score990.pkl', help='A dictionary where the key is the pathway and the value is the genes belonging to the pathway')
    parser.add_argument('--exp', default='./Data/Cell/CCLE_2369_EXP.csv', help='A csv file where row is cell line, column is gene, and value is the gene expression value')
    parser.add_argument('--ppi', default='./Data/Cell/CCLE_2369_PPI_990.csv', help='A csv file where row and column are gene, and value is 1 if there is an edge and 0 otherwise')
    
    return parser.parse_args()

def save_cell_graph(args):
    with open(args.kegg, 'rb') as file:
        kegg = pickle.load(file)
    
    exp = pd.read_csv(args.exp, index_col=0)
    index = exp.index
    columns = exp.columns

    scaler = StandardScaler()
    exp = scaler.fit_transform(exp)
    with open(os.path.join(args.path_data, 'scaler.pkl'), 'wb') as file:
        pickle.dump(scaler, file)

    imp_mean = SimpleImputer()
    exp = imp_mean.fit_transform(exp)
    with open(os.path.join(args.path_data, 'imp_mean.pkl'), 'wb') as file:
        pickle.dump(imp_mean, file)

    exp = pd.DataFrame(exp, index=index, columns=columns)
    cell_names = exp.index

    cell_dict = {}

    genes = exp.columns.to_list()
    for i in tqdm((cell_names)):

        # disjoint graph (with pathway)
        x_mask = []
        x = []
        gene_list = {}
        for p, pw in enumerate(list(kegg)):
            gene_list[pw] = []
            for gene in kegg[pw]:
                if gene in genes:
                    gene_list[pw].append(gene)
                    x_mask.append(p)
            x.append(exp.loc[i, gene_list[pw]])
        x = pd.concat(x)
        cell_dict[i] = Data(x=torch.tensor([x], dtype=torch.float).T, x_mask=torch.tensor(x_mask, dtype=torch.int))

    # print(cell_dict)
    with open(os.path.join(args.path_data, f'cell_feature_std.pkl'), 'wb') as file:
              pickle.dump(cell_dict, file)
    print("finish saving cell data!")
    
    return gene_list


def get_STRING_edges(args, gene_list):

    new_gene_list = []
    for genes in gene_list.values:
        new_gene_list.extend(genes)
    gene_list = new_gene_list
    
    save_path = os.path.join(args.path_data, f'edge_index.npy')
    if not os.path.exists(save_path):
        # gene_list
        ppi = pd.read_csv(args.ppi, index_col=0)

        # joint graph (without pathway)
        ppi = ppi.loc[gene_list, gene_list].values
        sparse_mx = sparse.csr_matrix(ppi).tocoo().astype(np.float32)
        edge_index = np.vstack((sparse_mx.row, sparse_mx.col))

        # Conserve edge_index
        # print(len(edge_index[0]))
        np.save(
            save_path,
            edge_index)


if __name__ == '__main__':
    args = parse_args()

    genelist = save_cell_graph(args)
    get_STRING_edges(args, gene_list=genelist)
