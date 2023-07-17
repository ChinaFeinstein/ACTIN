import torch
from torch_geometric.data import InMemoryDataset,Data
import random
import networkx as nx
from pysmiles import read_smiles
import pandas as pd
#---progromed by feinstein  Version:1.0.0 ----------
#-------lastest data:2022.12.1---------------------
#---调用方法--dataset=MyOwnDataset(root='')----------------------------------------------

class MyPreDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        print('dataset has loaded')
        return ['pre2.das']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        file=input('drug file path:')
 
        data_drug = pd.read_excel(str(file))

        gene = pd.read_csv('E:\my\my_gene.csv')
        diff = pd.read_excel('E:\my\DGE-978.xlsx')
        lookup_tab = ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'I', 'P', 'Au', 'B', 'Ca', 'Si', 'Hg']
  
        data_list = []
        for i,row in data_drug.iterrows():
            smiles=row['SMILES']
            x,edge_index,num_x=smiles_encoder(smiles,lookup_tab)
            for q in range(483):
                gene_name=diff['gene_symbol'][q]
                gene_value =diff['logFC'][q]
                gene_vetor =torch.tensor(gene[gene_name].to_list(),dtype=float).view(1,-1)

                y = torch.tensor(gene_value).view(1,-1)
                data = Data(x=x,edge_index=edge_index,y=y,gene=gene_vetor)
                    
                data_list.append(data)
      
        print('-'*100)
        print(lookup_tab)
        print('-'*100)
   

        #random.shuffle(data_list)

  
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


#---------------------处理SMILES数据到图数据（x，edge_index）------------------------------------------

def smiles_encoder(smiles:str,lookup_tab):  #lookup_tab 是一个列表['C','N','O',....]
    print(smiles)
    mol = read_smiles(smiles)
    elements = nx.get_node_attributes(mol, name = "element")
    atom_list=list(elements.values())
    tab=torch.zeros([len(atom_list),len(lookup_tab)])
    for i, atom in enumerate(atom_list):
        index=lookup_tab.index(atom)
        if index!=None:
            tab[i,index]=1
        if index==None:
            print('can not find atom',atom)
    
    edge_index=torch.tensor(list(mol.edges),dtype=torch.long)
    edge_index=edge_index.t().contiguous()
    a = torch.flip(edge_index, [0])
    edge_index=torch.cat((edge_index,a),dim=1)
    
    
    return tab, edge_index,len(atom_list)

if __name__=='__main__':
    dataset = MyPreDataset(root='E:/data')

    print(dataset.len())

#----------------------------获取lookup_tab--------------------------------------


