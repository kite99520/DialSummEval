import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import re
from collections import defaultdict
import pandas as pd

model_type_dict = {
    'GOLD': 'A',
    'LONGEST_3': 'B', 'LEAD_3': 'C',
    'PGN': 'D', 'Transformer' : 'E',
    'BART': 'F', 'PEGASUS': 'G', 'UniLM': 'H',
    'CODS': 'I','ConvoSumm': 'J', 'MV_BART': 'K', 'PLM': 'L', 'PNEP': 'M', 'S_BART': 'N'
}

# summ_home_path = '/home/gaomq/SAMSUM/human_ann/processed/models'
summ_home_path = './reproduce/analysis/models_eval_new'

# source_path = '/home/gaomq/SAMSUM/human_ann/processed/models/dials.txt'
source_path = './reproduce/analysis/models_eval_new/dials.txt'

# ref_path = './reproduce/analysis/models_eval_new/A/summs.txt'
ref_path = './reproduce/analysis/models_eval_new/A/summs.txt'


def print_moverscore_score(output_fname='moverscore.csv'):
    from moverscore_v2 import get_idf_dict, word_mover_score 

    for k,v in tqdm(model_type_dict.items()):
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        refs = list(map(lambda x:x.strip(),open(ref_path).readlines()))
    
        idf_dict_hyp = get_idf_dict(summaries) # idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = get_idf_dict(refs) # idf_dict_ref = defaultdict(lambda: 1.)

        scores = word_mover_score(refs, summaries, idf_dict_ref, idf_dict_hyp, \
                            stop_words=[], n_gram=1, remove_subwords=True)

        data_dict = {'moverscore':scores}
        output_each_path = os.path.join(summ_home_path, v, output_fname)
        df = pd.DataFrame(data_dict)
        df.to_csv(output_each_path, index=False)

if __name__ == '__main__':
    print_moverscore_score()