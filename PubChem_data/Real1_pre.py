import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import csv


filename = 'AID_1346986_datatable.csv'
data = pd.read_csv(filename)
simplified = pd.DataFrame()
simplified['PUBCHEM_SID']=data['PUBCHEM_SID']
simplified['PUBCHEM_CID']=data['PUBCHEM_CID']
simplified['activity'] = data['PUBCHEM_ACTIVITY_SCORE']

"""
myist = simplified['PUBCHEM_SID'].to_list()[1:]

with open('lookup_data.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     for word in mylist:
        try:
            wr.writerow([int(word)])
        except:
            print(word)
"""
chem_data = Chem.SDMolSupplier('ChemRecords.sdf')

radius = 2
nBits = 512

y = [] 
ECFP6 = [] 
missed = 0 
acts = simplified['activity'].to_list()[1:]

for i in range(len(acts)):
    try:
        ECFP6.append(AllChem.GetMorganFingerprintAsBitVect(chem_data[i],radius=radius, nBits=nBits))
        y.append(acts[i])
    except:
        missed += 1

ecfp6_name = [f'Bit_{i}' for i in range(nBits)]
ecfp6_bits = [list(l) for l in ECFP6]
df_morgan = pd.DataFrame(ecfp6_bits, columns=ecfp6_name)

df_morgan.to_csv('Real1_fingerprints.csv')

activity = pd.DataFrame(y)
activity.to_csv('Real1_targets.csv')

