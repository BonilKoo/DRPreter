import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class CellEncoder(nn.Module):
    def __init__(self, num_feature, num_genes, layer_cell, dim_cell):
        super().__init__()
        self.num_feature = num_feature
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.final_node = num_genes
        self.convs_cell = nn.ModuleList()

        for i in range(self.layer_cell):
            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
            else:
                conv = GATConv(self.num_feature, self.dim_cell)

            self.convs_cell.append(conv)

  
    def forward(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))

        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        return node_representation
    
     
    def grad_cam(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            if i == 0:
                cell_node = cell.x
                cell_node.retain_grad()
                
        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        return cell_node, node_representation