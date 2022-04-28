import json
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import nltk
from collections import defaultdict
import pandas as pd
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

# ref_path = './reproduce/analysis/models_eval_new/A/summs.txt'
ref_path = './reproduce/analysis/models_eval_new/A/summs.txt'

dae_data_path = '/home/gaomq/factuality-datasets/data_to_eval'

def create_eval_jsonl(summary_path, out_jsonl_path):
    documents = list(map(lambda x:x.replace(' | ',' ').strip('|').strip(),open(source_path).readlines()))
    # documents = list(map(lambda x:x.strip('|').strip(),open(source_path).readlines()))
    summaries = list(map(lambda x:x.strip(),open(summary_path).readlines()))

    # fname = 'data-dev.jsonl'

    # the label is useless, in model evaluation
    # bart_file_path = '/home/gaomq/factCC/data_to_eval/bart'
    # fpath = os.path.join(bart_file_path,fname)
    fpath = out_jsonl_path
    with open(fpath,'w') as f:
        for idx, (text, claims) in enumerate(zip(documents, summaries)):
            claims = nltk.sent_tokenize(claims)
            for claim in claims:
                data = {'id':idx, 'text':text, 'claim':claim, 'label':'CORRECT'}
                f.write(json.dumps(data,ensure_ascii='False')+'\n')

def run_eval(input_file, output_file):

    cmd = '''
        python eval_my.py \
        --model_type electra_dae \
        --model_dir /home/gaomq/factuality-datasets/ENT-C_dae  \
        --input_file {} \
        --output_file {} \
    '''.format(input_file, output_file)
    os.system(cmd)



def get_dae_score(src_jsonl_file,eval_json_file):
    
    guids = []
    with open(src_jsonl_file) as f:
        line = f.readline().strip()
        while line:
            d = json.loads(line)
            guids.append(d['id'])
            line = f.readline().strip()

    res = json.load(open(eval_json_file))
    
    id2acc = dict()
    for guid, label in zip(guids,res):
        if guid not in id2acc:
            id2acc[guid] = [label]
        else:
            id2acc[guid].append(label)
    accs = []
    for k, v in id2acc.items():
        accs.append(v.count(0)/len(v))  # 0 indicates correct (faithful)
    return accs
    # print('average accuracy: {}'.format(np.mean(accs)))


def print_dae_score(output_fname):
    for k,v in tqdm(model_type_dict.items()):
        
        fpath = os.path.join(summ_home_path, v, 'summs.txt')

        model_path = os.path.join(dae_data_path, v)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        jsonl_path = os.path.join(model_path, 'data-dev.jsonl')
        eval_json_path = os.path.join(model_path, 'eval_model_result.json')

        create_eval_jsonl(fpath, jsonl_path)
        run_eval(input_file=jsonl_path, output_file=eval_json_path)
        scores = get_dae_score(src_jsonl_file=jsonl_path,eval_json_file=eval_json_path)
        
        data_dict = {'dae':scores}
        output_each_path = os.path.join(summ_home_path, v, output_fname)
        df = pd.DataFrame(data_dict)
        df.to_csv(output_each_path, index=False)


if __name__ == '__main__':
    print_dae_score(output_fname='dae.csv')