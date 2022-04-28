import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import re
from collections import defaultdict
import pandas as pd
import torch


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


def convert2csv(input_fname):
    data = defaultdict(list)
    with open(input_fname) as f:
        for line in f:
            w_line = line.strip().split('\t')
            for i, seg in enumerate(w_line):
                if i == 0:
                    data['model_type'].append(seg.strip(':'))
                else:
                    metric_name, value =  seg.split(':')
                    value = float(value)
                    data[metric_name].append(value)
    df = pd.DataFrame(data)
    outpath = os.path.splitext(input_fname)[0] + '.csv'
    df.to_csv(outpath, index=False)      


def print_summaqa_score(output_fname='summaqa.csv'):
    # from summaqa import evaluate_corpus
    from summaqa import QG_masked, QA_Metric

    question_generator = QG_masked()
    qa_metric = QA_Metric()

    # out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        documents = list(map(lambda x:x.replace(' | ',' ').strip('|').strip(),open(source_path).readlines()))
        prob = []
        fscore = []
        for summ, docu in zip(summaries, documents):
            masked_questions, answer_spans = question_generator.get_questions(docu)
            score_1 = qa_metric.compute(masked_questions, answer_spans, summ)
            prob.append(score_1['avg_prob'])
            fscore.append(score_1['avg_fscore'])
        data_dict = {'summaqa_prob':prob, 'summaqa_fscore': fscore}
        output_each_path = os.path.join(summ_home_path, v, output_fname)
        df = pd.DataFrame(data_dict)
        df.to_csv(output_each_path, index=False)
        # res = evaluate_corpus(documents, summaries)
        # out_f.write('{}:\tsumma_p:{}\tsumma_f:{}\n'.format(k,res['avg_prob'], res['avg_fscore']))
    # out_f.close()

def print_nlgeval_score(output_fname='nlgeval.csv'):
    # from summaqa import evaluate_corpus
    from nlgeval import NLGEval
    nlgeval = NLGEval(no_skipthoughts=True)  # loads the models

    # out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        references = list(map(lambda x:x.strip(),open(ref_path).readlines()))
        
        data_dict = defaultdict(list)

        for hypo, ref in zip(summaries, references):
            metrics_dict = nlgeval.compute_individual_metrics(ref=[ref], hyp=hypo)
            for m, val in metrics_dict.items():
                if m != 'EmbeddingAverageCosineSimilairty' and m != 'ROUGE_L' and m != 'CIDEr':
                    data_dict[m].append(val)
        
        output_each_path = os.path.join(summ_home_path, v, output_fname)
        df = pd.DataFrame(data_dict)
        df.to_csv(output_each_path, index=False)

def print_questeval_score(output_fname='questeval.csv'):
    from questeval.questeval_metric import QuestEval
    questeval = QuestEval(no_cuda=True)
    
    # out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        documents = list(map(lambda x:x.replace(' | ',' ').strip('|').strip(),open(source_path).readlines()))
        score = questeval.corpus_questeval(hypothesis=summaries, sources=documents)
        output_each_path = os.path.join(summ_home_path, v, output_fname)

        data_dict = {'questeval':score['ex_level_scores']}
        output_each_path = os.path.join(summ_home_path, v, output_fname)
        df = pd.DataFrame(data_dict)
        df.to_csv(output_each_path, index=False)



def print_rouge_score(output_fname='rouge.csv'):
    import rouge
    # apply_avg = True
    apply_avg = False
    apply_best = False
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                        max_n=4,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=apply_avg,
                        apply_best=apply_best,
                        alpha=0.5,  # Default F1_score
                        weight_factor=1.2,
                        stemming=True)

    # out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        # documents = list(map(lambda x:x.replace(' | ',' ').strip('|').strip(),open(source_path).readlines()))
        refs = list(map(lambda x:x.strip(),open(ref_path).readlines()))
        
        scores = evaluator.get_scores(summaries, refs)
        output_each_path = os.path.join(summ_home_path, v, output_fname)


        m_types = ['rouge-1','rouge-2','rouge-3','rouge-4','rouge-l']
        df_dict = {}
        for m in m_types:
            df_dict[m] = [s['f'][0] for s in scores[m]]
        df = pd.DataFrame(df_dict)
        df.to_csv(output_each_path,index=False)


def print_blanc_score(output_fname='blanc.csv'):
    from blanc import BlancHelp, BlancTune

    blanc_help = BlancHelp(device='cuda', inference_batch_size=128)
    # blanc_help = BlancHelp(device='cpu', inference_batch_size=128)
    blanc_tune = BlancTune(device='cuda', inference_batch_size=24, finetune_mask_evenly=False, finetune_batch_size=24)
    # blanc_tune = BlancTune(device='cpu', inference_batch_size=24, finetune_mask_evenly=False, finetune_batch_size=24)
    
    # out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        documents = list(map(lambda x:x.replace(' | ',' ').strip(),open(source_path).readlines()))
        b_help = blanc_help.eval_pairs(documents, summaries)
        b_tune = blanc_tune.eval_pairs(documents, summaries)

        data_dict = {'blanc_help':b_help,'blanc_tune':b_tune}
        output_each_path = os.path.join(summ_home_path, v, output_fname)
        df = pd.DataFrame(data_dict)
        df.to_csv(output_each_path, index=False)


def print_bertscore_score(output_fname='bertscore.csv'):
    from bert_score import score

    # out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        refs = list(map(lambda x:x.strip(),open(ref_path).readlines()))

        P, R, F1 = score(summaries, refs, lang="en")
        
        data_dict = {'bertscore_p':P,'bertscore_r':R,'bertscore_f1':F1}
        output_each_path = os.path.join(summ_home_path, v, output_fname)
        df = pd.DataFrame(data_dict)
        df.to_csv(output_each_path, index=False)



def ppl(text):

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = "cuda"
    model_id = "gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    encodings = tokenizer(text, return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl



def print_ppl_score(output_fname='ppl.csv'):

    # out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        # refs = list(map(lambda x:x.strip(),open(ref_path).readlines()))

        # P, R, F1 = score(summaries, refs, lang="en")
        res = [ppl(summ) for summ in summaries]

        data_dict = {'ppl':res}
        output_each_path = os.path.join(summ_home_path, v, output_fname)
        df = pd.DataFrame(data_dict)
        df.to_csv(output_each_path, index=False)




def print_human_sys_score(output_fname):
    out_f = open(output_fname,'w')
    for k,v in tqdm(model_type_dict.items()):
        
        rpath = os.path.join(summ_home_path, v, 'ratings.txt')
        scores = {'coherance':[],'consistency':[],'fluency':[],'relevance':[]}
        with open(rpath) as f:
            for line in f:
                raw_scores = [ list(map(lambda x:float(x),s.split(','))) for s in line.strip().split('\t')]
                scores['coherance'].append(np.mean(raw_scores[0]))
                scores['consistency'].append(np.mean(raw_scores[1]))
                scores['fluency'].append(np.mean(raw_scores[2]))
                scores['relevance'].append(np.mean(raw_scores[3]))

        out_f.write('{}:\tcoherance:{:.4f}\tconsistency:{:.4f}\tfluency:{:.4f}\trelevance:{:.4f}\n'.format(k,np.mean(scores['coherance']),np.mean(scores['consistency']),np.mean(scores['fluency']),np.mean(scores['relevance'])))
    out_f.close()

  



if __name__ == '__main__':
    # print_summaqa_score('summaqa.csv')
    # print_nlgeval_score('nlgeval.csv')
    # print_questeval_score('questeval.txt')
    # print_rouge_score(output_fname='rouge_default.csv')
    # print_rouge_score(output_fname='rouge_2.txt',m_type='rouge-2')
    # print_rouge_score(output_fname='rouge_l.txt',m_type='rouge-l')
    
    # print_blanc_score('/home/gaomq/SAMSUM/human_ann/processed/metric_eval/blanc.txt')
    # print_human_sys_score('/home/gaomq/SAMSUM/human_ann/processed/metric_eval/human.txt')
    
    # for fname in os.listdir(metric_path):
    #     convert2csv(os.path.join(metric_path, fname))
    # compute_human_correlation('summaqa')
    pass
