import os
import numpy as np
from tqdm import tqdm
import pandas as pd



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_type_dict = {
    'GOLD': 'A',
    'LONGEST_3': 'B', 'LEAD_3': 'C',
    'PGN': 'D', 'Transformer' : 'E',
    'BART': 'F', 'PEGASUS': 'G', 'UniLM': 'H',
    'CODS': 'I','ConvoSumm': 'J', 'MV_BART': 'K', 'PLM': 'L', 'PNEP': 'M', 'S_BART': 'N'
}

summ_home_path = './reproduce/analysis/models_eval_new'

source_path = './reproduce/analysis/models_eval_new/dials.txt'

ref_path = './reproduce/analysis/models_eval_new/A/summs.txt'



def print_feqa_score(output_fname='feqa.csv'):
    from feqa import FEQA
    scorer = FEQA(use_gpu=False)
    # out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        documents = list(map(lambda x:x.replace(' | ',' ').strip(),open(source_path).readlines()))
        res = scorer.compute_score(documents, summaries, aggregate=False)
        output_each_path = os.path.join(summ_home_path, v, os.path.basename(output_fname))
        df_dict = {'feqa_score':res}
        df = pd.DataFrame(df_dict)
        df.to_csv(output_each_path,index=False)

if __name__ == '__main__':
    print_feqa_score()