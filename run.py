# coding=utf-8
import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.utils.data import DataLoader

from datasets import load_datasets_and_vocabs
from model import Aspect_CS_GAT_BERT
from trainer import train

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_name', type=str, default='rest16',
                        choices=['rest15', 'rest16'],
                        help='Choose absa dataset.')
    parser.add_argument('--output_dir', type=str, default='./data/output-gcn',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')


    parser.add_argument('--cuda_id', type=str, default='0',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')

    # Model parameters
    parser.add_argument('--bert_model_dir', type=str, default='./uncased_L-12_H-768_A-12/',
                        help='Directory storing BERT embeddings')

    parser.add_argument('--lower',  type= bool, default=True,
                        help='Sequences lower.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")


    parser.add_argument('--max_len', type=int, default=128,
                        help="Sequences max length.")
    parser.add_argument('--target_method', type=str, default='TO',
                        choices=['TO', 'BIO'],
                        help='Choose target encode method.')

    # Embedding
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')
    parser.add_argument('--wemb_dp', type=float, default=0.5,
                        help="Word embedding dropout value.")
    parser.add_argument('--posemb_dim', type=int, default=50,
                        help="Pos_class embedding dimension value.")
    parser.add_argument('--posemb_dp', type=float, default=0.5,
                        help="Pos_class embedding dropout value.")

    # Bi-LSTM
    parser.add_argument('--lstm_hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers of bilstm or highway or elmo.')
    parser.add_argument('--bilstm_dp', type=float, default=0.5,
                        help='Dropout rate for Bi-LSTM.')

    # GCN
    parser.add_argument('--gcn_dp', type=float, default=0.5,
                        help='Dropout rate for GCN.')
    parser.add_argument('--gcn_num_layers', type=int, default=2,
                        help='Number of layers of gcn.')
    parser.add_argument('--gcn_use_bn', type=bool, default=True,
                        help='Batch normalization for gcn.')

    # Highway
    parser.add_argument('--highway_use', type=bool, default=True,
                        help='Use highway.')

    # Attention
    parser.add_argument('--sa_hidden_size', type=int, default=300,
                        help='Hidden size of self-attention.')

    # Output
    parser.add_argument('--sa_classes', type=int, default=4,
                        help='Number of classes of sentiment analysis.')
    parser.add_argument('--output_dp', type=float, default=0.5,
                        help='Dropout rate for output.')

    # GAT
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of Attention heads.')

    return parser.parse_args()

def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Parse args
    args = parse_args()
    logger.info(vars(args))

    # Setup CUDA, GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    args.tokenizer = tokenizer

    # Load datasets and vocabs
    train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab= load_datasets_and_vocabs(args)

    model = Aspect_CS_GAT_BERT(args)

    model.to(args.device)
    # Train
    train(args, train_dataset, model, test_dataset, word_vocab)


if __name__ == "__main__":
    main()

