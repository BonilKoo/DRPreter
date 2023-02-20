import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import *
from Model.DRPreter import DRPreter
from Model.Similarity import Similarity
from torch_scatter import scatter_add


def arg_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--layer', type=int, default=3, help='Number of cell layers')
    parser.add_argument('--hidden_dim', type=int, default=8, help='hidden dim for cell')
    parser.add_argument('--layer_drug', type=int, default=3, help='Number of drug layers')
    parser.add_argument('--dim_drug', type=int, default=128, help='hidden dim for drug (default: 128)')
    parser.add_argument('--dim_drug_cell', type=int, default=256, help='hidden dim for drug and cell (default: 256)')
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='Dropout ratio (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=300, help='Maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping (default: 10)')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--path_result', default='./Result', help='A path to save results')
    
    parser.add_argument('--edge_index', default='./Data/Cell/edge_index.npy', help='An edge index file generated from cellline_graph.py')
    parser.add_argument('--dataset', default='./Data/sorted_IC50_82833_580_170.csv', help='Dataset containing information for cell line, drug, and IC50 value')
    parser.add_argument('--drug_dict', default='./Data/Drug/drug_feature_graph.pkl', help='A dictionary file generated from drug_graph.py')
    parser.add_argument('--cell_dict', default='./Data/Cell/cell_feature_std.pkl', help='A dictionary file generated from cellline_graph.py')
    parser.add_argument('--model_file', default=None, help='File name for saving/loading pth model')
    parser.add_argument('--kegg', default='./Data/Cell/34pathway_score990.pkl', help='A dictionary which was used for cellline_graph.py')
    
    return parser.parse_args()


def main():
    args = arg_parse()
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    result_path = args.path_result
    os.makedirs(result_path, exist_ok=True)
    
    print(f'seed: {args.seed}')
    set_random_seed(args.seed)
    
    edge_index = np.load(args.edge_index)
    
    data = pd.read_csv(args.dataset)
    
    with open(args.drug_dict, 'rb') as file:
        drug_dict = pickle.load(file) # pyg format of drug graph
    with open(args.cell_dict, 'rb') as file:
        cell_dict = pickle.load(file) # pyg data format of cell graph

    for key in cell_dict.keys():
        example_cell = cell_dict[key]
        args.num_cell_feature = example_cell.x.shape[1] # 1
        args.num_genes = example_cell.x.shape[0] # 4646
        break
    for key in drug_dict.keys():
        example_drug = drug_dict[key]
        args.num_drug_feature = example_drug.x.shape[1] # 77
        break
    # print(f'num_cell_feature: {args.num_feature}, num_genes: {args.num_genes}')
            
    gene_list = scatter_add(torch.ones_like(example_cell.x.squeeze()), example_cell.x_mask.to(torch.int64)).to(torch.int)
    args.max_gene = gene_list.max().item()
    args.cum_num_nodes = torch.cat([gene_list.new_zeros(1), gene_list.cumsum(dim=0)], dim=0)
    args.n_pathways = gene_list.size(0)
    print(f'num_genes:{args.num_genes}, num_edges:{len(edge_index[0])}')
    print(f'gene distribution: {gene_list}')
    print(f'mean degree:{(len(edge_index[0]) / args.num_genes):.4f}')
        
    
    train_loader, val_loader, test_loader = load_data(data, drug_dict, cell_dict, torch.tensor(edge_index, dtype=torch.long), args)
    print(f'total: {len(data)}, train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}')

    model = DRPreter(args).to(args.device)
    # print(model)

        
# -----------------------------------------------------------------
            
            
    if args.mode == 'train':
        result_col = ('mse\trmse\tmae\tpcc\tscc')
        
        result_prefix = 'results'
        results_path = get_path(args, result_path, result_prefix=f'val_{result_prefix}')
        
        with open(results_path, 'w') as f:
            f.write(result_col + '\n')
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        os.makedirs(f'weights', exist_ok=True)
        if args.model_file is None:
            state_dict_name = f'weights/weight_seed{args.seed}.pth'
        else:
            state_dict_name = args.model_file
        stopper = EarlyStopping(mode='lower', patience=args.patience, filename=state_dict_name)

        for epoch in range(1, args.epochs + 1):
            print(f"===== Epoch {epoch} =====")
            train_loss = train(model, train_loader, criterion, opt, args)

            mse, rmse, mae, pcc, scc, _ = validate(model, val_loader, args)
            results = [epoch, mse, rmse, mae, pcc, scc]
            save_results(results, results_path)
            
            print(f"Validation mse: {mse:.4f}")
            test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, df = validate(model, test_loader, args)
            print(f"Test mse: {test_MSE:.4f}")
            early_stop = stopper.step(mse, model)
            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        print(f'Best epoch: {epoch-stopper.counter}')

        stopper.load_checkpoint(model)

        train_MSE, train_RMSE, train_MAE, train_PCC, train_SCC, _ = validate(model, train_loader, args)
        val_MSE, val_RMSE, val_MAE, val_PCC, val_SCC, _ = validate(model, val_loader, args)
        test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, df = validate(model, test_loader, args)

        print('-------- DRPreter -------')
        print(f'##### Seed: {args.seed} #####')
        print('\t\tMSE\tRMSE\tMAE\tPCC\tSCC')
        print(f'Train result:\t{train_MSE:.4f}\t{train_RMSE:.4f}\t{train_MAE:.4f}\t{train_PCC:.4f}\t{train_SCC:.4f}')
        print(f'Val result:\t{val_MSE:.4f}\t{val_RMSE:.4f}\t{val_MAE:.4f}\t{val_PCC:.4f}\t{val_SCC:.4f}')
        print(f'Test result:\t{test_MSE:.4f}\t{test_RMSE:.4f}\t{test_MAE:.4f}\t{test_PCC:.4f}\t{test_SCC:.4f}')
        df.to_csv(get_path(args, result_path, result_prefix=f'test{result_prefix}_df', extension='csv'), sep='\t', index=0)

        
    elif args.mode == 'test':
        if args.model_file is None:
            state_dict_name = f'weights/weight_seed{args.seed}.pth'
        else:
            state_dict_name = args.model_file
        model.load_state_dict(torch.load(state_dict_name, map_location=args.device)['model_state_dict'])


#         '''Get embeddings of specific drug and cell line pair'''
#         drug_name, cell_name = 'Bortezomib', 'ACH-000137' # 8MGBA
#         drug_emb, cell_emb = embedding(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
#         print(drug_emb, cell_emb)
        
        
        ''' Test results only '''
        test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, df = validate(model, test_loader, args)
        print('-------- DRPreter -------')
        print(f'##### Seed: {args.seed} #####')
        print('\t\tMSE\tRMSE\tMAE\tPCC\tSCC')
        print(f'Test result:\t{test_MSE:.4f}\t{test_RMSE:.4f}\t{test_MAE:.4f}\t{test_PCC:.4f}\t{test_SCC:.4f}')
        

        '''Interpolation of unknown values'''
        inference(model, drug_dict, cell_dict, edge_index, f'{result_path}/inference_seed{args.seed}.xlsx', args, data)
        
        
        '''GradCAM'''
        # ----- (1) Calculate gradient-based importance score for one cell line-drug pair -----        
        # drug_name, cell_name = 'Dihydrorotenone', 'ACH-001374'
        # os.makedirs(f'GradCAM')
        # gradcam_path =  get_path(args, 'GradCAM/', result_prefix=f'{drug_name}_{cell_name}_gradcam', extension='csv')
        
        # with open('Data/Cell/cell_idx2gene_dict.pkl', 'rb') as file:
        #     gene_dict = pickle.load(file)
        
        # # Save importance score
        # sorted_cell_node_importance, indices = gradcam(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
        # idx2gene = [gene_dict[idx] for idx in indices]
        
        # sorted_cell_node_importance = list(sorted_cell_node_importance.detach().cpu().numpy())
        # indice = list(indices)
        
        # df = pd.DataFrame((zip(sorted_cell_node_importance, indice, idx2gene)), columns=['cell_node_importance','indice','idx2gene'])
        # # df.to_csv(gradcam_path, index=False)
        # print(*list(df['idx2gene'])[:30])
        
        # ----- (2) Calculate scores from total test set in 'inference.csv' -----
        data = pd.read_excel(f'{result_path}/inference_seed{args.seed}.xlsx', sheet_name='test')
        name = data[['Drug name', 'DepMap_ID']]
        
        with open('Data/Cell/cell_idx2gene_dict.pkl', 'rb') as file:
            gene_dict = pickle.load(file)
        
        total_gene_df = pd.Series(list(range(len(data))))
        for i in tqdm(range(len(data))):
            drug_name, cell_name = name.iloc[i]
            _, indices = gradcam(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
            idx2gene = [gene_dict[idx] for idx in indices]
            gene_df = pd.DataFrame(idx2gene)
            total_gene_df.loc[i] = ', '.join(list(gene_df.drop_duplicates(keep='first')[0])[:5])
        
        data['Top5 genes'] = total_gene_df
        data.to_excel(f'{result_path}/inference_seed{args.seed}_gradcam.xlsx', sheet_name='test')
        
            
        '''Visualize pathway-drug self-attention score from Transformer'''
        # ----- (1) For one cell line - drug pair -----
        
        drug_name, cell_name = 'Dasatinib', 'ACH-000072'

        # # print(cell_name)
        attn_score = attention_score(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
        # print(f'attn_score: {attn_score}')
        # print(f'attn_score.shape: {attn_score.shape}') # attn_score.shape: torch.Size([1, 35, 35])
        # # print(torch.sum(attn_score, axis=1))
        with open(args.kegg, 'rb') as file:
            pathway_names = pickle.load(file).keys()
        tks = [p[5:] for p in list(pathway_names)]
        tks.append(drug_name)
        # # print(tks)
        draw_pair_heatmap(attn_score, drug_name, cell_name, tks, args)
        
        # ----- (2) Heatmap of all cell lines of one drug -----
        
        # drug_name = 'Rapamycin'
        # data = pd.read_csv(f'./Data/{drug_name}.csv')
        # cell_list = list(data['DepMap_ID'])
        
        # result_dict = {}
        # total_result = np.full(35, 0)
        # for cell_name in tqdm(cell_list):
        #     attn_score = attention_score(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
        #     print(attn_score.shape)
        #     attn_score = torch.squeeze(attn_score).cpu().detach().numpy()
        #     print(np.sum(attn_score, axis=1))
        #     result_dict[cell_name] = attn_score[-1, :] # (35, 1)
        #     total_result = np.vstack([total_result, attn_score[-1, :]])
        
        # with open('Data/Cell/34pathway_score990.pkl', 'rb') as file:
        #     pathway_names = pickle.load(file).keys()
        # xtks = [p[5:] for p in list(pathway_names)]
        # xtks.append(drug_name)
        # total_result = total_result[1:,:-1]
        # draw_drug_heatmap(total_result, drug_name, xtks, cell_list, args)
        
        
if __name__ == "__main__":
    main()