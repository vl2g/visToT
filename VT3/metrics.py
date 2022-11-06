# %%
import os
import warnings
import json

import nltk.translate.bleu_score as bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk.translate.meteor_score as meteor
from rouge import Rouge

import argparse
import numpy as np

warnings.filterwarnings("ignore")

def read_txt(path):
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(line.strip('\n'))
    return data

chencherry = SmoothingFunction()
def eval_generated_text(predictions_path, reference_path, image_data_path):
    '''
    Evaluate predicted senteces against reference sentences using
    BLEU, METEOR and ROUGE metrics
    Note:
        Edited from table-only code to adapt for table+image data.
        Here we evaluate generated texts for all images associated with
        a given table, and only keep the text that has maximum score 
        across most all metrics to count towards final score.

    '''
    predictions = read_txt(predictions_path)
    references = read_txt(reference_path)
    image_data = json.load(open(image_data_path))
    table_idxs = []
    for table_idx, images in enumerate(image_data):
        for image in images:
            table_idxs.append(table_idx)
    n_tables = len(image_data)
    references = [references[table_idx] for table_idx in table_idxs]

    tokenized_references = [[ref.split(' ')] for ref in references]
    tokenized_predictions = [pred.split(' ') for pred in predictions]

    scores = {
        'bleu': [],
        'meteor': [],
        'rouge-1': [],
        'rouge-2': [],
        'rouge-l': [],
    }

    avg_scores_all = dict.fromkeys(scores)

    rouge = Rouge()
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        if len(pred) == 0:
            score = {'rouge-1':{'f': 0.0}, 'rouge-2':{'f': 0.0}, 'rouge-l':{'f': 0.0}}
        else:
            score = rouge.get_scores([pred], [ref], avg=False)[0]
        rouge_scores.append(score)

    for rouge_metric in ['rouge-1', 'rouge-2', 'rouge-l']:
        scores[rouge_metric] = list(map(lambda score: score[rouge_metric]['f'], rouge_scores))

    for reference, prediction in zip(tokenized_references, tokenized_predictions):
        bleu_score = bleu.sentence_bleu(reference, prediction, smoothing_function=chencherry.method1)
        scores['bleu'].append(bleu_score)
        meteor_score = meteor.meteor_score(reference, prediction)
        scores['meteor'].append(meteor_score)

    for metric in scores.keys():
        avg_scores_all[metric] = np.array(scores[metric]).mean() * 100
    
    print("--Results--")
    print("BLEU: {:.2f}\nMETEOR: {:.2f}\nROUGE-1: {:.2f}\nROUGE-2: {:.2f}\nROUGE-l: {:.2f}".format(
      avg_scores_all['bleu'],
      avg_scores_all['meteor'],
      avg_scores_all['rouge-1'],
      avg_scores_all['rouge-2'],
      avg_scores_all['rouge-l'],
    ))

    return avg_scores_all['bleu']

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str)
    parser.add_argument('--references', type=str)
    parser.add_argument('--image_data', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    if not os.path.exists(args.predictions):
        raise ValueError('Incorrect predictions path: {}'.format(args.predictions))
    if not os.path.exists(args.references):
        raise ValueError('Incorrect references path: {}'.format(args.references))
    if not os.path.exists(args.image_data):
        raise ValueError('Incorrect image data path: {}'.format(args.image_data))

    eval_generated_text(args.predictions, args.references, args.image_data)