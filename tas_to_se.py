import argparse
import json
import os
import re
import sys
import copy

from nltk.tokenize import TreebankWordTokenizer
import xml.etree.ElementTree as ET
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor

data_path = './data/experiment_datas'

def convert(in_file_name, out_file_name, predictor):
    with open(in_file_name, 'r') as fr, open(out_file_name, 'w') as fw:
        title = fr.readline()
        sentence_lines = fr.readlines()
        fw.write('\t'.join(['sentence_id', 'category', 'sentence', 'sentiment', 'ner_tags']))
        fw.write('\n')
        for i in range(0, len(sentence_lines), 3):
            sent_flag = False
            for j in range(3):
                _, yes_no, category_sentiment, text, target_tags = sentence_lines[i+j].split('\t')
                category = ' '.join(category_sentiment.split()[:2])
                if yes_no == '1':
                    fw.write(_ + '\t' + category + '\t' + text + '\t' + category_sentiment.split()[-1] + '\t' + target_tags.strip() + '\n')
                    sent_flag = True
            if sent_flag == False:
                fw.write(_ + '\t' + category + '\t' + text + '\t' + 'N/A' + '\t' + target_tags.strip() + '\n')

    json_data = []
    idx = 0

    with open(out_file_name, 'r') as fr:
        title = fr.readline()
        prev_text = None
        for sentence in tqdm(fr.readlines()):
            _, category, text, sentiment, target_tags = sentence.split('\t')
            if prev_text != text:
                doc = predictor.predict(text)
                prev_text = text

            example = dict()
            example["sentence"] = text
            example['tokens'] = doc['words']
            example['tags'] = doc['pos']
            example['predicted_dependencies'] = doc['predicted_dependencies']
            example['predicted_heads'] = doc['predicted_heads']

            example['dependencies'] = []
            predicted_dependencies = doc['predicted_dependencies']
            predicted_heads = doc['predicted_heads']
            for idx, item in enumerate(predicted_dependencies):
                dep_tag = item
                frm = predicted_heads[idx]
                to = idx + 1
                example['dependencies'].append([dep_tag, frm, to])

            example['sentiment'] = sentiment
            example['category'] = category
            example['target_tags'] = target_tags.strip().split(' ')


            text_split = text.split(' ')
            len_text = len(text_split)
            tt_bias = 0
            index = 0
            while len_text != len(example['tokens']):
                if index > (len(text_split)-1):
                    break
                if text_split[index] == example['tokens'][index+tt_bias]:
                    index += 1
                    continue
                else:
                    bias_nb = 0
                    merge_word = ''
                    while True:
                        merge_word += example['tokens'][index+tt_bias+bias_nb]
                        if merge_word == text_split[index]:
                            break
                        else:
                            bias_nb += 1
                    if example['target_tags'][index+tt_bias] in ['T', 'O', 'I']:
                        for i in range(bias_nb):
                            example['target_tags'].insert(index+tt_bias, example['target_tags'][index+tt_bias])
                    else:
                        for i in range(bias_nb):
                            example['target_tags'].insert(index+tt_bias+1, 'I')
                    tt_bias += bias_nb
                    index += 1
                len_text += tt_bias

            assert (len(example['target_tags']) == len(example['tokens']))
            json_data.append(example)

    extended_filename = out_file_name.replace('.tsv', '_biaffine_depparsed.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))

def main():
    model_path = ("./data/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
    data = [('train_15_TAS.tsv', 'test_15_TAS.tsv'),
            ('train_16_TAS.tsv', 'test_16_TAS.tsv')]
    predictor = Predictor.from_path(model_path)
    for train_file, test_file in data:
        convert(os.path.join(data_path, 'BIO', train_file), os.path.join(data_path, 'BIO', 'cs_' + train_file), predictor)
        convert(os.path.join(data_path, 'BIO', test_file), os.path.join(data_path, 'BIO', 'cs_' + test_file), predictor)
        convert(os.path.join(data_path, 'TO', train_file), os.path.join(data_path, 'TO', 'cs_' + train_file), predictor)
        convert(os.path.join(data_path, 'TO', test_file), os.path.join(data_path, 'TO', 'cs_' + test_file), predictor)

if __name__ == "__main__":
    main()
