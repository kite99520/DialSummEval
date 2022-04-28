from bart_score import BARTScorer
import numpy as np
import os
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


def print_bartscore(output_fname='bartscore.csv'):
    from bart_score import BARTScorer
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='bart.pth')
    # out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        documents = list(map(lambda x:x.replace(' | ',' ').strip('|').strip(),open(source_path).readlines()))
        references = list(map(lambda x:x.strip(),open(ref_path).readlines()))
        res_s_h = bart_scorer.score(documents, summaries)
        res_h = bart_scorer.score(['' for i in range(len(summaries))], summaries)
        res_h_r = bart_scorer.score(summaries, references)
        res_r_h = bart_scorer.score(references, summaries)
        
        output_each_path = os.path.join(summ_home_path, v, output_fname)

        df_dict = {'bartscore_s_h':res_s_h,'bartscore_h':res_h,'bartscore_h_r':res_h_r,'bartscore_r_h':res_r_h}
        df = pd.DataFrame(df_dict)
        df.to_csv(output_each_path,index=False)



if __name__ == '__main__':
    print_bartscore('bartscore.csv')