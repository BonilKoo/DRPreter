import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import pickle
import math
from Model.DRPreter import DRPreter
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import os


def save_results(results, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, results)) + '\n')
        

def get_path(args, result_path='', result_prefix='results', extension='txt'):
    path = f'{result_path}/{result_prefix}_seed{args.seed}.{extension}'
    return path


def train(model, loader, loss_fn, opt, args):
    model.train()
    device = args.device

    for data in tqdm(loader, desc='Iteration'):
        drug, cell, label = data
        drug, cell, label = drug.to(device), cell.to(device), label.to(device)
        output = model(drug, cell)
        loss = loss_fn(output, label.view(-1, 1).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f'Train Loss: {loss:.4f}')

    return loss


def validate(model, loader, args):
    model.eval()
    device = args.device
    
    y_true = [] 
    y_pred = [] 

    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Iteration'):
            drug, cell, label = data
            drug, cell, label = drug.to(device), cell.to(device), label.to(device)
            output = model(drug, cell)
            total_loss += F.mse_loss(output, label.view(-1, 1).float(), reduction='sum')
            y_true.append(label.view(-1, 1))
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    df = np.array([y_pred.squeeze().detach().cpu().numpy(), y_true.squeeze().detach().cpu().numpy()])
    df = pd.DataFrame(df.T, columns=['y_pred','y_true'])
    
    mse = (total_loss / len(loader.dataset)).detach().cpu().numpy()
    rmse = (torch.sqrt(total_loss / len(loader.dataset))).detach().cpu().numpy()
    mae = mean_absolute_error(y_true.detach().cpu(), y_pred.detach().cpu())
    pcc = pearsonr(y_true.detach().cpu().numpy().flatten(), y_pred.detach().cpu().numpy().flatten())[0]
    scc = spearmanr(y_true.detach().cpu().numpy().flatten(), y_pred.detach().cpu().numpy().flatten())[0]
    return mse, rmse, mae, pcc, scc, df


def gradcam(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args):
    cell_dict[cell_name].edge_index = torch.tensor(edge_index, dtype=torch.long)
    drug = Batch.from_data_list([drug_dict[drug_name]]).to(args.device)
    cell = Batch.from_data_list([cell_dict[cell_name]]).to(args.device)
    
    model.eval()
    
    drug_representation = model.DrugEncoder(drug)
    drug_representation = model.drug_emb(drug_representation)

    cell_node, cell_representation = model.CellEncoder.grad_cam(cell) # cell node: torch.Size([4646, 8]), cell_representation: torch.Size([1, 37168])
    # print(f'cell node: {cell_node.shape}, cell_representation: {cell_representation.shape}')
    mask = cell.x_mask[cell.batch==0].to(torch.long)
    cell_representation = model.cell_emb(model.padding(cell_representation, mask))

    # combine drug feature and cell line feature
    x, _ = model.aggregate(cell_representation, drug_representation) # x.shape: torch.Size([1, 512])
    ic50 = model.regression(x)
    ic50.backward()
    
    cell_node_importance = torch.relu((cell_node*torch.mean(cell_node.grad, dim=0)).sum(dim=1)) # for regression task
    # cell_node_importance = torch.abs((cell_node * torch.mean(cell_node.grad, dim=0)).sum(dim=1)) # for classification task
    cell_node_importance = cell_node_importance / cell_node_importance.sum()
    sorted, indices = torch.sort(cell_node_importance, descending=True)

    return sorted, indices.detach().cpu().numpy()


def embedding(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args):
    cell_dict[cell_name].edge_index = torch.tensor(edge_index, dtype=torch.long)
    drug = Batch.from_data_list([drug_dict[drug_name]]).to(args.device)
    cell = Batch.from_data_list([cell_dict[cell_name]]).to(args.device)
    
    model.eval()
    drug_representation, cell_representation = model._embedding(drug, cell)

    return drug_representation, cell_representation


def attention_score(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args):
    cell_dict[cell_name].edge_index = torch.tensor(edge_index, dtype=torch.long)
    drug = Batch.from_data_list([drug_dict[drug_name]]).to(args.device)
    cell = Batch.from_data_list([cell_dict[cell_name]]).to(args.device)
    
    model.eval()
    score = model.attn_score(drug, cell)[0]

    return score



def draw_pair_heatmap(attn_score, drug_name, cell_name, ticks, args):
    attn_score = torch.squeeze(attn_score).mean(axis=0).detach().cpu().numpy()
    # print(np.sum(attn_score, axis=1))
    # print(attn_score)
    # attn_score = np.flip(attn_score, axis=0)
    # print(attn_score)
   
    # df = pd.DataFrame(attn_score)
    # sns.heatmap(attn_score, annot=True, fmt='.1f')
    # ax = sns.heatmap(attn_score, xticklabels=ticks, yticklabels=yticks)
    ax = sns.heatmap(attn_score, cmap='Reds')

    # plt.ylim(0, len(attn_score)
    ax.invert_yaxis()
    ax.set_xticks(range(len(ticks)))
    ax.set_yticks(range(len(ticks)))
    ax.set_xticklabels(ticks, fontsize=6)
    ax.set_yticklabels(ticks, fontsize=6) # Due to 'invert_yaxis', yticklables can be flipped without 'ticks.reverse()'
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment("bottom")
    
    # cellname_dict = np.load('./Data/Cell/cell_depmap2name_dict.npy', allow_pickle=True)
    # # print(cell_name)
    # cell_name = cellname_dict[cell_name]
    plt.title(f'{drug_name} - {cell_name} self attention score')
    os.makedirs(f'Result/Heatmap', exist_ok=True)
    plt.savefig('Result/' f'Heatmap/seed{args.seed}_{drug_name}_{cell_name}.png')



def draw_drug_heatmap(attn_score, drug_name, xticks, yticks, args):
    # sns.set(rc = {'figure.figsize':(5,10)})
    ax = sns.heatmap(attn_score, cmap='Reds')

    ax.set_xticks(range(len(xticks)))
    ax.set_yticks(range(len(yticks)))
    ax.set_xticklabels(xticks, fontsize=6)
    ax.set_yticklabels(yticks, fontsize=4)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment("bottom")
        
    # cellname_dict = np.load('./Data/Cell/cell_depmap2name_dict.npy', allow_pickle=True)
    # # print(cell_name)
    # cell_name = cellname_dict[cell_name]
    plt.title(f'{drug_name} self attention score')
    os.makedirs(f'Result/Heatmap', exist_ok=True)
    plt.savefig('Result/' + f'Heatmap/seed{args.seed}_{drug_name}.png')
    
    

def inference(model, drug_dict, cell_dict, edge_index, save_name, args, IC):
    """
    Predict missing values
    """    
    model.eval()

    train_set, val_test_set = train_test_split(IC, test_size=0.2, random_state=args.seed)
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=args.seed)
        
    cell_table = IC[["DepMap_ID", "stripped_cell_line_name"]].drop_duplicates(keep='first')
    drug_table = IC["Drug name"].drop_duplicates(keep='first').to_frame() # to_frame(): Convert series to dataframe
    cell_table['value'] = 1 # Temporary variable for obtaining every combination of cell and drug according to the value
    drug_table['value'] = 1
    drug_cell_table = drug_table.merge(cell_table, how='left', on='value') # 98,600 combination
    del drug_cell_table['value']
    '''
    All drug-cellline combinations are stacked on top, and only pairs with IC50 values ​​are stacked on the bottom.
    In drop_duplicate, if you delete duplicates from the top and bottom (= those with IC50) with keep=False, only unknown pairs remain
    '''
    unknown_set = pd.concat([drug_cell_table, IC[["Drug name", "DepMap_ID", "stripped_cell_line_name"]]], axis=0)
    unknown_set.drop_duplicates(keep=False, inplace=True) # Pairs with no IC50 values
    dataset = {'train':train_set, 'val':val_set, 'test':test_set, 'unknown':unknown_set}
    writer = pd.ExcelWriter(save_name)
    for dataset_name, data in dataset.items():
        data.reset_index(drop=True, inplace=True)
        IC50_pred = []
        with torch.no_grad():
            drug_name, cell_ID, cell_line_name = data['Drug name'], data["DepMap_ID"], data["stripped_cell_line_name"]
            for cell in cell_ID:
                cell_dict[cell].edge_index = torch.tensor(edge_index, dtype=torch.long)
            drug_list = [drug_dict[name] for name in drug_name]
            cell_list = [cell_dict[name] for name in cell_ID]
            batch_size = 2048
            batch_num = math.ceil(len(drug_list)/batch_size)
            for index in range(batch_num):
                drug = Batch.from_data_list(drug_list[index*batch_size:(index+1)*batch_size]).to(args.device)
                cell = Batch.from_data_list(cell_list[index*batch_size:(index+1)*batch_size]).to(args.device)
                y_pred = model(drug, cell)
                IC50_pred.append(y_pred)
            IC50_pred = torch.cat(IC50_pred, dim=0)
        table = pd.concat([drug_name, cell_ID, cell_line_name], axis=1)
        if dataset_name != 'unknown':
            table["IC50"] = data["IC50"]
        table["IC50_Pred"] = IC50_pred.detach().cpu().numpy()
        if dataset_name != 'unknown':
            table["Abs_error"] = np.abs(IC50_pred.detach().cpu().numpy()-np.array(table["IC50"]).reshape(-1,1))
        table.to_excel(writer, sheet_name=dataset_name, index=False)
        torch.cuda.empty_cache()
    writer.close()

        
        
class MyDataset(Dataset):
    def __init__(self, drug_dict, cell_dict, IC, edge_index):
        super(MyDataset, self).__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True) # Discard old indexes after train_test_split and rearrange with the new indexes
        self.drug_name = IC['Drug name']
        self.Cell_line_name = IC['DepMap_ID']
        self.value = IC['IC50']
        # self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_index = edge_index

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        self.cell[self.Cell_line_name[index]].edge_index = self.edge_index
        # self.cell[self.Cell_line_name[index]].adj_t = SparseTensor(row=self.edge_index[0], col=self.edge_index[1])
        return (self.drug[self.drug_name[index]], self.cell[self.Cell_line_name[index]], self.value[index])


def _collate(samples):
    drugs, cells, labels = map(list, zip(*samples))
    batched_drug = Batch.from_data_list(drugs)
    batched_cell = Batch.from_data_list(cells)
    return batched_drug, batched_cell, torch.tensor(labels)


def load_data(IC, drug_dict, cell_dict, edge_index, args, val_ratio=0.1, test_ratio=0.1):
    
    Dataset = MyDataset
    collate_fn = _collate
    
    if test_ratio == 1:
        train_loader, val_loader = None, None
        test_dataset = Dataset(drug_dict, cell_dict, IC, edge_index=edge_index)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
    else:
        train_set, val_test_set = train_test_split(IC, test_size=val_ratio+test_ratio, random_state=args.seed)
        val_set, test_set = train_test_split(val_test_set, test_size=test_ratio/(val_ratio+test_ratio), random_state=args.seed)    

        train_dataset = Dataset(drug_dict, cell_dict, train_set, edge_index=edge_index)
        val_dataset = Dataset(drug_dict, cell_dict, val_set, edge_index=edge_index)
        test_dataset = Dataset(drug_dict, cell_dict, test_set, edge_index=edge_index)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


class EarlyStopping():
    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        """
        Args:
            mode (str):   'higher': Higher metric suggests a better model / 'lower': Lower metric suggests a better model
            patience (int): The early stopping will happen if we do not observe performance improvement for 'patience' consecutive epochs.
            filename (str, optional): Filename for storing the model checkpoint. 
                                                  If not specified, it will automatically generate a file starting with 'early_stop' based on the current time.
            metric (str, optional):  A metric name that can be used to identify if a higher value is better, or vice versa.
        """
        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                f"'rmse' or 'roc_auc_score', got {metric}"
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score', 'accuracy']:
                print(f'For metric {metric}, the higher the better')
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print(f'For metric {metric}, the lower the better')
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """
        Check if the new score is higher than the previous best score.

        Args:
            score (float): New score.
            prev_best_score (float): Previous best score.

        Returns:
            (bool): Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """
        Check if the new score is lower than the previous best score.

        Args:
            score (float): New score.
            prev_best_score (float):  Previous best score.

        Returns:
            (bool): Whether the new score is lower than the previous best score.
        """ 
        return score < prev_best_score

    def step(self, score, model):
        """
        Update based on a new score.
        The new score is typically model performance on the validation set for a new epoch.

        Args:
            score (float): New score
            model (nn.Module): Model instance

        Returns:
            self.early_stop (bool): Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        """
        Saves model when the metric on the validation set gets improved.

        Args:
            model (nn.Module): Model instance.
        """        
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        """
        Load the latest checkpoint

        Args:
            model (nn.Module): Model instance.
        """        
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])



def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def boxplot():
    """
    Draw a boxplot sorted in descending order based on median of predicted IC50 values for each drug
    """
    data = pd.read_csv('Data/sorted_IC50_82833_580_170.csv')
    ic50 = data[['Drug name', 'IC50']]
    grouped = ic50.groupby('Drug name') # Grouping data by drug name
    df = pd.DataFrame({col:vals['IC50'] for col,vals in grouped})
    meds = df.median()
    meds.sort_values(ascending=False, inplace=True)
    df = df[meds.index]
    df.boxplot(fontsize='small', figsize=(100,20))
    plt.tick_params(axis='x', labelrotation=90)
    plt.show()