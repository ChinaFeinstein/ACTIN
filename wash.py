import pandas as pd
from pysmiles import read_smiles
import networkx as nx

file=input("file path: ")
data_drug = pd.read_excel(str(file))

diff=[]
num=[]

for i,row in data_drug.iterrows():
    smiles=row['SMILES']
    mol = read_smiles(smiles)
    elements = nx.get_node_attributes(mol, name = "element")
    atom_list=list(elements.values())
    lookup_tab = ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'I', 'P', 'Au', 'B', 'Ca', 'Si', 'Hg']
    D = set(atom_list).difference(set(lookup_tab)) #差集，在list2中但不在list1中的元素
    if len(D) !=0:
        diff.append(D)
        num.append(smiles)
        id=i

print(diff)
print(num)
print(id+2)