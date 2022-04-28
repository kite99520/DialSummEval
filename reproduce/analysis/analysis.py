import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import re
from collections import defaultdict
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from scipy.stats import pearsonr
import math
import matplotlib.pyplot as plt
import seaborn as sns



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

def merge_csv(metrics, out_fname='metrics.csv'):
    for k, v in model_type_dict.items():
        dfs = []
        for m in metrics:
            input_csv = os.path.join(summ_home_path, v, m+'.csv')
            dfs.append(pd.read_csv(input_csv))
        df = pd.concat(dfs, axis=1)

        out_path = os.path.join(summ_home_path, v, out_fname)
        df.to_csv(out_path, index=False)    


def get_human_scores(fname, human_rate_type):
    '''
        return numpy.array (100, 14)
    '''
    raw_scores = []
    for k,v in model_type_dict.items():
        df = pd.read_excel(fname, sheet_name=v)
        raw_scores.append(df[human_rate_type].tolist())
    return np.array(raw_scores).T


def get_metric_scores(metric_name, metrics_file='metrics.csv'):
    '''
        return numpy.array (100, 14)
    '''
    raw_scores = []
    for k,v in model_type_dict.items():
        total_metric_path = os.path.join(summ_home_path, v, metrics_file)
        df = pd.read_csv(total_metric_path)
        raw_scores.append(df[metric_name].tolist())
    return np.array(raw_scores).T       

def filter_noise_scores(s_1, s_2, s_3):
    '''
        filter noise ratings of human and give the average value
        return numpy.array (100, 14)
    '''
    res = np.zeros((100, 14))
    s_1 = s_1.tolist()
    s_2 = s_2.tolist()
    s_3 = s_3.tolist()
    for i, (r_1, r_2, r_3) in enumerate(zip(s_1, s_2, s_3)):
        for j, (a, b, c) in enumerate(zip(r_1, r_2, r_3)):
            if a == b and a != c:
                res[i,j] = a
            elif a == c and a != b:
                res[i,j] = c
            elif b == c and a != b:
                res[i,j] = b
            else:
                res[i,j] = (a + b + c) / 3
    return res


def cal_pearsonr(human_rate_type, metric_names=None, level='summary'):

    # use average to compute human scores
    # get origin scores for each annotator, dimension: (100, 14)
    human_scores_1 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_1.xlsx',
    human_rate_type=human_rate_type)

    human_scores_2 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_2.xlsx',
    human_rate_type=human_rate_type)

    human_scores_3 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_3.xlsx',
    human_rate_type=human_rate_type)

    # human_scores_avg = (human_scores_1 + human_scores_2 + human_scores_3) / 3  #  (100, 14)
    human_scores_avg = filter_noise_scores(human_scores_1, human_scores_2, human_scores_3) #  (100, 14)
    human_scores_avg_sys = np.mean(human_scores_avg, axis=0)  # (14,)
    # print(human_scores_avg_sys.shape)
    human_scores_avg = human_scores_avg.tolist()

    if not metric_names:
        metric_names = pd.read_csv('./reproduce/analysis/models_eval_new/A/metrics.csv').columns.tolist()
        # metric_names = pd.read_csv('./reproduce/analysis/models_eval_new/A/bartscore_my_all.csv').columns.tolist()
        # metric_names = sorted(metric_names, key=lambda x:x[::-1])
    # get the scores of the specific metric
    for metric in metric_names:
        corelations = []  # scores for dialogues (100, )
        metric_scores = get_metric_scores(metric)  # (100, 14)
        # metric_scores = get_metric_scores(metric, 'bartscore_my_all.csv')  # (100, 14)
        metric_scores_sys = np.mean(metric_scores, axis=0) # (14,)
        metric_scores = metric_scores.tolist()  # (100, 14)
        for human_, metric_ in zip(human_scores_avg, metric_scores):
            corr, p_value = pearsonr(human_, metric_)
            # corr, p_value = spearmanr(human_, metric_)
            # corr, p_value = kendalltau(human_, metric_)
            if not math.isnan(corr):
                corelations.append(corr)
        if level == 'summary':
            print('Summary Level {}\t{:<35}{:.4f}'.format(human_rate_type, metric, np.mean(corelations)))
        else:
            corr_sys, p_value_sys = pearsonr(human_scores_avg_sys, metric_scores_sys)
            # corr_sys, p_value_sys = spearmanr(human_scores_avg_sys, metric_scores_sys)
            # corr_sys, p_value_sys = kendalltau(human_scores_avg_sys, metric_scores_sys)
            print('System Level {}\t{:<35}{:.4f}\tp-value\t{:.4f}'.format(human_rate_type, metric, corr_sys, p_value_sys))


def print_human_ratings(human_rate_type):
    human_scores_1 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_1.xlsx',
    human_rate_type=human_rate_type)

    human_scores_2 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_2.xlsx',
    human_rate_type=human_rate_type)

    human_scores_3 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_3.xlsx',
    human_rate_type=human_rate_type)

    # human_scores_avg = (human_scores_1 + human_scores_2 + human_scores_3) / 3  #  (100, 14)
    human_scores_avg = filter_noise_scores(human_scores_1, human_scores_2, human_scores_3) #  (100, 14)
    human_scores_avg_sys = np.mean(human_scores_avg, axis=0)
    human_scores_avg_sys = human_scores_avg_sys.tolist()

    for k, s in zip(model_type_dict.keys(), human_scores_avg_sys):
        print('{}\t{:<15}\t{:.4f}'.format(human_rate_type, k, s))
        

def print_rouge_sample():

    for k, v in model_type_dict.items():
        file_path = os.path.join(summ_home_path, v, 'rouge_default.csv')
        df = pd.read_csv(file_path)
        print('{:<15}\tR-1 {:.4f}\tR-2 {:.4f}\tR-L {:.4f}'.format(k, df['rouge-1'].mean(), df['rouge-2'].mean(), df['rouge-l'].mean()))

        

def get_corr_matrix_metrics(figure_path):
    data_dict = dict()
    metric_names = pd.read_csv('./reproduce/analysis/models_eval_new/A/metrics.csv').columns.tolist()
    for metric in metric_names:
        corelations = []  # scores for dialogues (100, )
        metric_scores = get_metric_scores(metric)  # (100, 14)
        metric_scores_sys = np.mean(metric_scores, axis=0) # (14,)
        if metric == 'EmbeddingAverageCosineSimilarity':
            data_dict['Embedding Average'] = metric_scores_sys
        elif metric == 'VectorExtremaCosineSimilarity':
            data_dict['Vector Extrema'] = metric_scores_sys
        elif metric == 'GreedyMatchingScore':
            data_dict['Greedy Matching'] = metric_scores_sys
        else:
            data_dict[metric] = metric_scores_sys
        

    df = pd.DataFrame.from_dict(data_dict)
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.set(font_scale=1.0)
    plt.subplots(figsize=(14,15),dpi=1000)
    sns.heatmap(corr, mask=mask, annot=False, vmax=1, vmin=-1, square=True, cmap="Blues")
    plt.savefig(figure_path)
    
    
def get_corr_matrix_dimension(figure_path):
    data_dict = dict()
    dimensions = ['Coherence','Consistency','Fluency','Relevance']
    for human_rate_type in dimensions:

        human_scores_1 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_1.xlsx',
        human_rate_type=human_rate_type)

        human_scores_2 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_2.xlsx',
        human_rate_type=human_rate_type)

        human_scores_3 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_3.xlsx',
        human_rate_type=human_rate_type)

        # human_scores_avg = (human_scores_1 + human_scores_2 + human_scores_3) / 3  #  (100, 14)
        human_scores_avg = filter_noise_scores(human_scores_1, human_scores_2, human_scores_3) #  (100, 14)
        human_scores_avg_sys = np.mean(human_scores_avg, axis=0)  # (14,)
        data_dict[human_rate_type] = human_scores_avg_sys
        

    df = pd.DataFrame.from_dict(data_dict)
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.set(font_scale=1.5)
    plt.subplots(figsize=(14,15),dpi=400)
    sns.heatmap(corr, mask=mask, annot=True, vmax=1, vmin=-1, square=True, cmap="Greys")
    plt.savefig(figure_path)



def _is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass
    return False


def _plot_cal_pearsonr_bartscore(human_rate_type, metric_names=None, level='system'):

    # use average to compute human scores
    # get origin scores for each annotator, dimension: (100, 14)
    human_scores_1 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_1.xlsx',
    human_rate_type=human_rate_type)

    human_scores_2 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_2.xlsx',
    human_rate_type=human_rate_type)

    human_scores_3 = get_human_scores(fname='/home/gaomq/SAMSUM/human_ann_new/offline_ann_3.xlsx',
    human_rate_type=human_rate_type)

    # human_scores_avg = (human_scores_1 + human_scores_2 + human_scores_3) / 3  #  (100, 14)
    human_scores_avg = filter_noise_scores(human_scores_1, human_scores_2, human_scores_3) #  (100, 14)
    human_scores_avg_sys = np.mean(human_scores_avg, axis=0)  # (14,)
    # print(human_scores_avg_sys.shape)
    human_scores_avg = human_scores_avg.tolist()

    if not metric_names:
        # metric_names = pd.read_csv('./reproduce/analysis/models_eval_new/A/metrics.csv').columns.tolist()
        metric_names = pd.read_csv('./reproduce/analysis/models_eval_new/A/bartscore_my_all.csv').columns.tolist()
        metric_names = sorted(metric_names, key=lambda x:x[::-1])
    # get the scores of the specific 
    x = list(range(0,10000,1000))
    b_h = list(range(10))
    b_s_h = list(range(10))
    b_h_r = list(range(10))
    b_r_h = list(range(10))
    m_b_h = []
    m_b_s_h = []
    m_b_h_r = []
    m_b_r_h = []


    for metric in metric_names:
        corelations = []  # scores for dialogues (100, )
        metric_scores = get_metric_scores(metric, 'bartscore_my_all.csv')  # (100, 14)
        metric_scores_sys = np.mean(metric_scores, axis=0) # (14,)
        metric_scores = metric_scores.tolist()  # (100, 14)
        for human_, metric_ in zip(human_scores_avg, metric_scores):
            corr, p_value = pearsonr(human_, metric_)
            # corr, p_value = spearmanr(human_, metric_)
            # corr, p_value = kendalltau(human_, metric_)
            if not math.isnan(corr):
                corelations.append(corr)
        if level == 'summary':
            # print('Summary Level {}\t{:<35}{:.4f}'.format(human_rate_type, metric, np.mean(corelations)))
            corr_ = np.mean(corelations)
            ind = int(metric[10:14]) // 1000 if _is_number(metric[10:14]) else 0
            if metric.endswith('_s_h'):
                b_s_h[ind] = corr_
            elif metric.endswith('_h_r'):
                b_h_r[ind] = corr_
            elif metric.endswith('_r_h'):
                b_r_h[ind] = corr_
            else:
                b_h[ind] = corr_

        else:
            corr_sys, p_value_sys = pearsonr(human_scores_avg_sys, metric_scores_sys)
            # corr_sys, p_value_sys = spearmanr(human_scores_avg_sys, metric_scores_sys)
            # corr_sys, p_value_sys = kendalltau(human_scores_avg_sys, metric_scores_sys)
            # print('System Level {}\t{:<35}{:.4f}\tp-value\t{:.4f}'.format(human_rate_type, metric, corr_sys, p_value_sys))
            corr_ = corr_sys
            ind = int(metric[10:14]) // 1000 if _is_number(metric[10:14]) else 0
            if metric.endswith('_s_h'):
                b_s_h[ind] = corr_
                if p_value_sys <=0.05:
                    m_b_s_h.append(ind)
            elif metric.endswith('_h_r'):
                b_h_r[ind] = corr_
                if p_value_sys <=0.05:
                    m_b_h_r.append(ind)
            elif metric.endswith('_r_h'):
                b_r_h[ind] = corr_
                if p_value_sys <=0.05:
                    m_b_r_h.append(ind)
            else:
                b_h[ind] = corr_
                if p_value_sys <=0.05:
                    m_b_h.append(ind)
    
    plt.figure(dpi=400)

    plt.plot(x, b_s_h, marker='*',markevery=m_b_s_h)
    # plt.plot(x, b_h)
    plt.plot(x, b_h, marker='*',markevery=m_b_h)
    # plt.plot(x, b_h_r)
    plt.plot(x, b_h_r, marker='*',markevery=m_b_h_r)

    # plt.plot(x, b_r_h)
    plt.plot(x, b_r_h, marker='*',markevery=m_b_r_h)

    plt.legend(['bartscore_s_h', 'bartscore_h', 'bartscore_h_r', 'bartscore_r_h'], loc='lower right')
    plt.xlabel('fine-tuning steps')
    plt.ylabel('correlation')
    # trans = {'Coherence':'连贯性','Relevance':'相关性','Consistency':'一致性','Fluency':'流畅性'}
    plt.title(human_rate_type)

    plt.savefig('/home/gaomq/SAMSUM/other_figs/bartscore_{}.jpg'.format(human_rate_type))




if __name__ == '__main__':
    # cal_pearsonr('Relevance',level='system')
    # print_human_ratings('Coherence')
    # get_corr_matrix_metrics('/home/gaomq/SAMSUM/figs/metrics_corr.jpg')
    # get_corr_matrix_dimension('/home/gaomq/SAMSUM/figs/dimensions_corr.jpg')
    # print_rouge_sample()
    _plot_cal_pearsonr_bartscore('Coherence')
    _plot_cal_pearsonr_bartscore('Fluency')
    _plot_cal_pearsonr_bartscore('Consistency')
    _plot_cal_pearsonr_bartscore('Relevance')
