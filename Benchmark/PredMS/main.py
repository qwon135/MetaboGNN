import logging
import os
import time
import shutil
import copy
import pandas as pd
import sys
import sklearn
import warnings
import os
import glob
import numpy as np
import pandas as pd
from rdkit import RDLogger
import utils
from rdkit import Chem
from mordred import Calculator, descriptors
import warnings , os
import subprocess
import shutil

from joblib import dump, load

def read_smiles(filename):
    smiles_list = []
    with open(filename, 'r') as fp:
        for line in fp:
            smiles_list.append(line.strip())
    return np.asarray(smiles_list)

def calc_descriptor(smiles_list):
    calc = Calculator(descriptors, ignore_3D = True)
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) 
        if mol != None:
            mols.append(mol)
            
    for each_mol in mols:
        try:
            AllChem.EmbedMolecule(each_mol, randomSeed=42)
        except:
            pass
    
    df = calc.pandas(mols, nproc=1)
    new_df = df.select_dtypes(include=['float64', 'int'])
    return new_df

def predict_ms_using_rf(X, model_dir):
    model_file = model_dir+'/best_rf_model.pkl'
    loaded_model = load(model_file) 
    pred_result = loaded_model.predict_proba(X)
    return list(pred_result)

def predict_ms_rf(all_des_df, model_dir, target_descriptor_file):
    target_features = []
    with open(target_descriptor_file, 'r') as fp:
        for line in fp:
            target_features.append(line.strip())

    df = all_des_df[target_features]
    X = df.values
    
    predicted_results = predict_ms_using_rf(X, model_dir)
    return predicted_results

def main():
    warnings.filterwarnings(action='ignore')
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    warnings.filterwarnings('ignore')
    start = time.time()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    parser = utils.argument_parser()
    
    options = parser.parse_args()
    smiles_file = options.smiles_file
    output_dir = options.output_dir
    
    try:
        shutil.rmtree(output_dir)
    except:
        pass
    try:
        os.mkdir(output_dir)
    except:
        pass
    
    smiles_list = read_smiles(smiles_file)
    all_des_df = calc_descriptor(smiles_list)
    
    model_dir = './models/rf_model/'
    target_descriptor_file = './models/200_features.txt'
    
    human_results = predict_ms_rf(all_des_df, model_dir, target_descriptor_file)
    
    with open(output_dir+'/results.txt', 'w') as fp:
        fp.write('%s\t%s\n'%('Smiles', 'Probability'))
        for i in range(len(smiles_list)):
            fp.write('%s\t%s\n'%(smiles_list[i], human_results[i][1]))
    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))

if __name__ == '__main__':
    main()