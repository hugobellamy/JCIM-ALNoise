import pandas as pd
import warnings
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import numpy as np

warnings.filterwarnings(action='ignore')

filename = 'AID_1347160_datatable.csv'
data = pd.read_csv(filename)
data = data.set_index('PUBCHEM_CID')
"""
mylist = simplified.index.to_list()[1:]
with open('lookup_data.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     for word in mylist:
        try:
            wr.writerow([int(word)])
        except:
            print(word)
"""
chem_data = pd.read_csv('Structures_Pub1.csv')

chem_data = chem_data.set_index('cid')

mol_list = data.index.to_list()[1:]
mol_list = [x for x in mol_list if x == x]

act_dict = data['PUBCHEM_ACTIVITY_SCORE'].to_dict()

radius = 2
nBits = 1024

fingerprint_list = []
index = []
y = []
missed = 0 

for i in mol_list:
    smiles = chem_data['isosmiles'][i]
    mol = Chem.MolFromSmiles(smiles)
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,radius=radius, nBits=nBits))
    fingerprint_list.append(fp)
    index.append(i)
    n = act_dict[i]
    y.append(float(n))


col_name = [f'Bit_{i}' for i in range(nBits)]
col_bits = [list(l) for l in fingerprint_list]
fingerprints = pd.DataFrame(col_bits, columns=col_name, index=index)

activity = pd.DataFrame(y, index=index)

fingerprints.to_csv('Real1_fingerprints.csv')
activity.to_csv('Real1_targets.csv')
