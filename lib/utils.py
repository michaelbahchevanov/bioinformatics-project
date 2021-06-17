import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import PandasTools

np.random.seed(42)

class Error(Exception):
    """Base class for exceptions"""
    pass

class DataObjectError(Error):
    """DataObjectError base class"""
    pass
    

class DataObject(object):
    def __init__(self, data=None,):
        if data is None:
            raise DataObjectError('Provide data')
        elif not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
            raise DataObjectError(f'{type(data)} not a DataFrame or Series object')
        else:
            self.data = data
    
    def filter_non_smiles(self):
        if isinstance(self.data, pd.DataFrame):
            if 'standard_value' not in self.data.columns or 'canonical_smiles' not in self.data.columns:
                raise DataObjectError('Provide a DataFrame with the standard_value and canonical_smiles column names.')
            
            self.data = self.data[self.data.standard_value.notna()]
            self.data = self.data[self.data.canonical_smiles.notna()]
            return self
        if isinstance(self.data, pd.Series):
            if self.data.name == 'canonical_smiles':
                self.data = self.data[self.data.notna()]
                return self
            raise DataObjectError('Provide a Series with the name of canonical_smiles')
    
    def drop_duplicate_smiles(self):
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.drop_duplicates(['canonical_smiles'])
            return self
        if isinstance(self.data, pd.Series):
            if self.data.name == 'canonical_smiles':
                self.data = self.data.drop_duplicates(keep='first')
                return self
    
    def classify_activity(self):
        def _label_activity(row):
            if float(row) >= 10000:
                return "inactive"
            elif float(row) <= 1000:
                return "active"
            else:
                return "intermediate"
        
        if isinstance(self.data, pd.Series):
            raise DataObjectError('Provide a Dataframe')
        if isinstance(self.data, pd.DataFrame):
            self.data['class'] = self.data.standard_value.apply(_label_activity)
            return self
        
    def save(self, version='1', name='temp', index=False):
        from pathlib import Path
        ROOT_DIR = str(Path(__file__).parent.parent)
        PATH = f'{ROOT_DIR}/data/{name}_{version}_bioactivity_data_preprocessed.csv'
        self.data.to_csv(PATH, index=index)
        return self
        
    def generate_lipinski_descriptors(self):
        molecule_data = [Chem.MolFromSmiles(element) for element in self.data.canonical_smiles]
    
        baseData = np.arange(1,1)
        
        for i, mol in enumerate(molecule_data):
            # molecular mass
            desc_MolWt = Descriptors.MolWt(mol)
            # octanol-water partition coefficient
            desc_MolLogP = Descriptors.MolLogP(mol)
            # number of hydrogen donors
            desc_NumHDonors = Lipinski.NumHDonors(mol)
            # number of hydrogen acceptors
            desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
            
            row = np.array([desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors])
            
            # handle first entry in the vector
            if i==0:
                baseData = row
            # stack all the other rows on the base data
            else:
                baseData = np.vstack([baseData, row])
                
        columnNames = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
        descriptors = pd.DataFrame(data=baseData, columns=columnNames)
        
        return pd.concat([self.data, descriptors], axis=1)
    
    def convert_to_neg_log(self, measurement='pEC50'):
        def normalize_values(row):
            if row > 100_000_000:
                return 100_000_000
            else:
                return row
            
        self.data.standard_value = self.data.standard_value.apply(normalize_values)
        self.data[measurement] = self.data.standard_value.apply(lambda x: -np.log10(x*(10 ** -9)))
        return self
    
    def calculate_mann_whitney(self, descriptor):
        from scipy.stats import mannwhitneyu
        
        if descriptor not in self.data.columns:
            raise DataObjectError(f'{descriptor} not in the DataFrame')
        
        active = self.data[self.data['class'] == 'active'][descriptor]
        inactive = self.data[self.data['class'] == 'inactive'][descriptor]
        
        # perform test on bioactivity classes
        stat, p = mannwhitneyu(active, inactive)
        
        # interpret test results
        alpha = 0.05
        if p > alpha:
            interpret = 'Same distribution (fail to reject H0)'
        else:
            interpret = 'Different distribution (reject H0)'
        
        results = pd.DataFrame({'Descriptor': descriptor,
                                'Statistics': stat,
                                'p': p,
                                'alpha': alpha,
                                'Interpretation': interpret
                            }, index=[0])
        
        save_path = f'../data/mannwhitneyu_{descriptor}_.csv'
        results.to_csv(save_path)
        return results
    
    def visualize_distribution(self, descriptor):
        if descriptor not in self.data.columns:
            raise DataObjectError(f'{descriptor} not in the DataFrame')
        
        import seaborn as sns
        sns.set(style='whitegrid')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(5.5, 5.5))

        sns.boxplot(x = 'class', y = f'{descriptor}', data = self.data)

        plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
        plt.ylabel(f'{descriptor}', fontsize=14, fontweight='bold')

        plt.savefig(f'../plots/{descriptor}_plot.pdf')
        
    def generate_morgan_matrix(self):
        morgan_matrix = np.zeros((1, 2048))
        l = len(self.data.canonical_smiles)
    
        # For each compound, get the structure, convert to Morgan fingerprint, and add to the morgan_matrix
        for i in range(l):
            try:
                compound = Chem.MolFromSmiles(self.data.canonical_smiles[i])
                fingerprint = Chem.AllChem.GetMorganFingerprintAsBitVect(compound, 2, nBits=2048)
                fingerprint = fingerprint.ToBitString()
                row = np.array([int(x) for x in list(fingerprint)])
                morgan_matrix = np.row_stack((morgan_matrix, row))
                
                # Progress checker
                if i % 500 == 0:
                    percentage = np.round(100*(i/l), 1)
                    print(f'{percentage}% done')
            except:
                print(f'problem at index: {i}')
        
        # Deleting the first row of zeros
        morgan_matrix = np.delete(morgan_matrix, 0, axis=0)
        
        print('\n')
        print(f'Morgan Matrix dimensions: {morgan_matrix.shape}')
        return morgan_matrix