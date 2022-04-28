import os
import pandas as pd
from tempfile import TemporaryDirectory
from tqdm import tqdm

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

ref_path = './reproduce/analysis/models_eval_new/A/summs.txt'


def create_tsv(refs_path, hypos_path, out_tsv_path):
    refs = list(map(lambda x:x.strip(),open(refs_path).readlines()))
    hypos = list(map(lambda x:x.strip(),open(hypos_path).readlines()))
    with open(out_tsv_path,'w') as f:
        for ref, hypo in zip(refs, hypos):
            f.write('{}\t{}\n'.format(ref, hypo))

def print_sms_score(output_fname='sms.csv'):

    for k,v in tqdm(model_type_dict.items()):
        with TemporaryDirectory() as dirname:
            fpath = os.path.join(summ_home_path, v, 'summs.txt')
            temp_tsv_path = os.path.join(dirname, 'test.tsv')
            create_tsv(refs_path=ref_path, hypos_path=fpath, out_tsv_path=temp_tsv_path)
            os.system('python smd.py {} glove sms'.format(temp_tsv_path))
            
            res_path = os.path.join(dirname,'test_glove_sms.out')
            df = pd.read_csv(res_path,sep='\t')
            scores = df['sms']
            output_each_path = os.path.join(summ_home_path, v, output_fname)
            scores.to_csv(output_each_path,index=False)
            


if __name__ == '__main__':
    print_sms_score()
