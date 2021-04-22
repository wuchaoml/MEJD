'''
Biaffine Dependency parser from AllenNLP
'''
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

model_path = ("./data/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Path to biaffine dependency parser.')
    parser.add_argument('--data_path', type=str, default='./data/experiment_datas',
                        help='Directory of where semeval14 or twiiter data held.')
    return parser.parse_args()

def syntaxInfo2json(file_path, predictor):
    json_data = []
    idx = 0

    with open(file_path, 'r') as fr:
        title = fr.readline()
        prev_text = None
        for sentence in tqdm(fr.readlines()):
            _, yes_no, category_sentiment, text, target_tags = sentence.split('\t')
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

            example['yes_no'] = yes_no
            example['category_sentiment'] = category_sentiment
            example['target_tags'] = target_tags.strip().split(' ')


            # Because
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

    extended_filename = file_path.replace('.tsv', '_biaffine_depparsed.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))


def main():
    args = parse_args()
    predictor = Predictor.from_path(args.model_path)

    # predictor = biaffine_parser_universal_dependencies_todzat_2017()

    # data = [('ABSA15_Restaurants_Train.xml', 'ABSA15_Restaurants_Test.xml'),
    #         ('ABSA16_Restaurants_Train.xml', 'ABSA15_Restaurants_Test.xml')]

    """
            ('BERT_15_category_train_TAS.txt', 'BERT_15_category_test_TAS.txt'),
            ('BERT_16_category_train_TAS.txt', 'BERT_16_category_test_TAS.txt')
    """
    data = [('train_15_TAS.tsv', 'test_15_TAS.tsv'),
            ('train_16_TAS.tsv', 'test_16_TAS.tsv')]
    for train_file, test_file in data:
        syntaxInfo2json(os.path.join(args.data_path, 'BIO', train_file), predictor)
        syntaxInfo2json(os.path.join(args.data_path, 'BIO', test_file), predictor)
        syntaxInfo2json(os.path.join(args.data_path, 'TO', train_file), predictor)
        syntaxInfo2json(os.path.join(args.data_path, 'TO', test_file), predictor)

if __name__ == "__main__":
    main()
