#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Maja Popovic

# The program is distributed under the terms 
# of the GNU General Public Licence (GPL)

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Publications of results obtained through the use of original or
# modified versions of the software have to cite the authors by refering
# to the following publication:

# Maja Popović (2015).
# "chrF: character n-gram F-score for automatic MT evaluation".
# In Proceedings of the Tenth Workshop on Statistical Machine Translation (WMT15), pages 392–395
# Lisbon, Portugal, September 2015.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import math
import unicodedata
import argparse
from collections import defaultdict
import time
import string

def separate_characters(line):
    return list(line.strip().replace(" ", ""))

def separate_punctuation(line):
    words = line.strip().split()
    tokenized = []
    for w in words:
        if len(w) == 1:
            tokenized.append(w)
        else:
            lastChar = w[-1] 
            firstChar = w[0]
            if lastChar in string.punctuation:
                tokenized += [w[:-1], lastChar]
            elif firstChar in string.punctuation:
                tokenized += [firstChar, w[1:]]
            else:
                tokenized.append(w)
    
    return tokenized
    
def ngram_counts(wordList, order):
    counts = defaultdict(lambda: defaultdict(float))
    nWords = len(wordList)
    for i in range(nWords):
        for j in range(1, order+1):
            if i+j <= nWords:
                ngram = tuple(wordList[i:i+j])
                counts[j-1][ngram]+=1
   
    return counts

def ngram_matches(ref_ngrams, hyp_ngrams):
    matchingNgramCount = defaultdict(float)
    totalRefNgramCount = defaultdict(float)
    totalHypNgramCount = defaultdict(float)
 
    for order in ref_ngrams:
        for ngram in hyp_ngrams[order]:
            totalHypNgramCount[order] += hyp_ngrams[order][ngram]
        for ngram in ref_ngrams[order]:
            totalRefNgramCount[order] += ref_ngrams[order][ngram]
            if ngram in hyp_ngrams[order]:
                matchingNgramCount[order] += min(ref_ngrams[order][ngram], hyp_ngrams[order][ngram])


    return matchingNgramCount, totalRefNgramCount, totalHypNgramCount


def ngram_precrecf(matching, reflen, hyplen, beta):
    ngramPrec = defaultdict(float)
    ngramRec = defaultdict(float)
    ngramF = defaultdict(float)
    
    factor = beta**2
    
    for order in matching:
        if hyplen[order] > 0:
            ngramPrec[order] = matching[order]/hyplen[order]
        else:
            ngramPrec[order] = 1e-16
        if reflen[order] > 0:
            ngramRec[order] = matching[order]/reflen[order]
        else:
            ngramRec[order] = 1e-16
        denom = factor*ngramPrec[order] + ngramRec[order]
        if denom > 0:
            ngramF[order] = (1+factor)*ngramPrec[order]*ngramRec[order] / denom
        else:
            ngramF[order] = 1e-16
            
    return ngramF, ngramRec, ngramPrec

def computeChrF(fpRef, fpHyp, nworder, ncorder, beta, sentence_level_scores = None):
    
    sent_chrf = []  # my_add
    norder = float(nworder + ncorder)

    # initialisation of document level scores
    totalMatchingCount = defaultdict(float)
    totalRefCount = defaultdict(float)
    totalHypCount = defaultdict(float)
    totalChrMatchingCount = defaultdict(float)
    totalChrRefCount = defaultdict(float)
    totalChrHypCount = defaultdict(float)
    averageTotalF = 0.0

    nsent = 0
    for hline, rline in zip(fpHyp, fpRef):
        nsent += 1
        
        # preparation for multiple references
        maxF = 0.0
        bestWordMatchingCount = None
        bestCharMatchingCount = None
        
        hypNgramCounts = ngram_counts(separate_punctuation(hline), nworder)
        hypChrNgramCounts = ngram_counts(separate_characters(hline), ncorder)

        # going through multiple references

        refs = rline.split("*#")

        for ref in refs:
            refNgramCounts = ngram_counts(separate_punctuation(ref), nworder)
            refChrNgramCounts = ngram_counts(separate_characters(ref), ncorder)

            # number of overlapping n-grams, total number of ref n-grams, total number of hyp n-grams
            matchingNgramCounts, totalRefNgramCount, totalHypNgramCount = ngram_matches(refNgramCounts, hypNgramCounts)
            matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount = ngram_matches(refChrNgramCounts, hypChrNgramCounts)
                    
            # n-gram f-scores, recalls and precisions
            ngramF, ngramRec, ngramPrec = ngram_precrecf(matchingNgramCounts, totalRefNgramCount, totalHypNgramCount, beta)
            chrNgramF, chrNgramRec, chrNgramPrec = ngram_precrecf(matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount, beta)

            sentRec  = (sum(chrNgramRec.values())  + sum(ngramRec.values()))  / norder
            sentPrec = (sum(chrNgramPrec.values()) + sum(ngramPrec.values())) / norder
            sentF    = (sum(chrNgramF.values())    + sum(ngramF.values()))    / norder

            if sentF > maxF:
                maxF = sentF
                bestMatchingCount = matchingNgramCounts
                bestRefCount = totalRefNgramCount
                bestHypCount = totalHypNgramCount
                bestChrMatchingCount = matchingChrNgramCounts
                bestChrRefCount = totalChrRefNgramCount
                bestChrHypCount = totalChrHypNgramCount
        # all the references are done


        # write sentence level scores
        # if sentence_level_scores:
        #   sentence_level_scores.write("%i::c%i+w%i-F%i\t%.4f\n"  % (nsent, ncorder, nworder, beta, 100*maxF))
        sent_chrf.append(maxF) # my_add

        # collect document level ngram counts
        for order in range(nworder):
            totalMatchingCount[order] += bestMatchingCount[order]
            totalRefCount[order] += bestRefCount[order]
            totalHypCount[order] += bestHypCount[order]
        for order in range(ncorder):
            totalChrMatchingCount[order] += bestChrMatchingCount[order]
            totalChrRefCount[order] += bestChrRefCount[order]
            totalChrHypCount[order] += bestChrHypCount[order]

        averageTotalF += maxF

    # all sentences are done
     
    # total precision, recall and F (aritmetic mean of all ngrams)
    totalNgramF, totalNgramRec, totalNgramPrec = ngram_precrecf(totalMatchingCount, totalRefCount, totalHypCount, beta)
    totalChrNgramF, totalChrNgramRec, totalChrNgramPrec = ngram_precrecf(totalChrMatchingCount, totalChrRefCount, totalChrHypCount, beta)

    totalF    = (sum(totalChrNgramF.values())    + sum(totalNgramF.values()))    / norder
    averageTotalF = averageTotalF / nsent
    totalRec  = (sum(totalChrNgramRec.values())  + sum(totalNgramRec.values()))  / norder
    totalPrec = (sum(totalChrNgramPrec.values()) + sum(totalNgramPrec.values())) / norder

    return totalF, averageTotalF, totalPrec, totalRec, sent_chrf


def main():

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

    # sys.stdout.write("start_time:\t%i\n" % (time.time()))


    argParser = argparse.ArgumentParser()
    # argParser.add_argument("-R", "--reference", help="reference translation", required=True)
    # argParser.add_argument("-H", "--hypothesis", help="hypothesis translation", required=True)
    argParser.add_argument("-nc", "--ncorder", help="character n-gram order (default=6)", type=int, default=6)
    argParser.add_argument("-nw", "--nworder", help="word n-gram order (default=2)", type=int, default=2)
    argParser.add_argument("-b", "--beta", help="beta parameter (default=2)", type=float, default=2.0)
    # argParser.add_argument("-s", "--sent", help="show sentence level scores", action="store_true")
    argParser.add_argument("-o", "--output_fname",type=str, default='chrf.csv')

    args = argParser.parse_args()


    for k,v in tqdm(model_type_dict.items()):
        
        fpath = os.path.join(summ_home_path, v, 'summs.txt')
        # summaries = list(map(lambda x:x.strip(),open(fpath).readlines()))
        # references = list(map(lambda x:x.strip(),open(ref_path).readlines()))
        rtxt = open(ref_path, 'r')
        htxt = open(fpath, 'r')
        totalF, averageTotalF, totalPrec, totalRec, sent_chrf = computeChrF(rtxt, htxt, args.nworder, args.ncorder, args.beta)
        data_dict = {'chrf':sent_chrf}
        
        output_each_path = os.path.join(summ_home_path, v, args.output_fname)
        df = pd.DataFrame(data_dict)
        df.to_csv(output_each_path, index=False)


if __name__ == "__main__":
    main()
