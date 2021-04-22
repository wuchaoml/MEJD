import logging
import os
import random
import collections
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm, trange
from transformers import AdamW
from optimization import BERTAdam

from datasets import my_collate
# from transformers import AdamW
# from transformers import BertTokenizer

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_input_from_batch(args, batch):
    inputs = {  'sentence_cs': batch[0],
                'pos_class': batch[1], # aspect token
                'text_len': batch[2], # reshaped
                'adjmv': batch[3],
                'yes_no': batch[4],
                'target_zo': batch[5],
                'input_mask': batch[6],
                'segment_type_ids': batch[7]
            }
    labels = None
    return inputs, labels

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def train(args, train_dataset, model, test_dataset, word_vocab):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size

    sampler_weights = [1 if data[6]==0 else 10 for data in train_dataset]
    train_sampler = WeightedRandomSampler(sampler_weights, len(train_dataset), replacement=True)
    # train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=my_collate)

    t_total = len(train_dataloader) * args.num_train_epochs

    # optimizer = get_bert_optimizer(args, model)
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
         ]

    num_train_steps = int(len(train_dataset) / args.per_gpu_train_batch_size * args.num_train_epochs)

    optimizer = BERTAdam(optimizer_parameters,
                     lr=args.learning_rate,
                     warmup=0.1,
                     t_total=num_train_steps)

    total_params_num, trainable_params_num = get_parameter_number(model)['Total'], get_parameter_number(model)['Trainable']

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Total params number = %d ", total_params_num)
    logger.info("  Total trainable params number = %d", trainable_params_num)

    result_dir = os.path.join(args.output_dir, args.dataset_name, args.target_method, time.strftime("%m-%d-%H-%M", time.localtime()))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    output_log_file = os.path.join(result_dir, "log.txt")
    print("output_log_file=",output_log_file)
    with open(output_log_file, "w") as writer:
        writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")


    global_step = 0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    epoch = 0
    for _ in train_iterator:
        epoch += 1
        tr_loss_sa, tr_loss_dt = 0.0, 0.0
        nb_tr_steps = 0
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs, _ = get_input_from_batch(args, batch)
            loss_sa, loss_dt, logits_sa, logits_dt, _  = model.forward(**inputs)

            loss_dt.backward(retain_graph=True)
            loss_sa.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss_sa += loss_sa.item()
            tr_loss_dt += loss_dt.item()
            nb_tr_steps += 1

            optimizer.step()
            model.zero_grad()
            global_step += 1

        test_sa_loss, test_dt_loss, test_accuracy = evaluate(args, test_dataset, model, epoch, word_vocab, result_dir)

        result = collections.OrderedDict()
        result = {'epoch': epoch,
                'global_step': global_step,
                'loss': tr_loss_sa/nb_tr_steps,
                'test_sa_loss': test_sa_loss,
                'test_dt_loss': test_dt_loss,
                'test_accuracy': test_accuracy}

        logger.info("***** Eval results *****")
        with open(output_log_file, "a+") as writer:
            for key in result.keys():
                logger.info("  %s = %s\n", key, str(result[key]))
                writer.write("%s\t" % (str(result[key])))
            writer.write("\n")

    return global_step, test_sa_loss/global_step, all_eval_results


def evaluate(args, test_dataset, model, epoch, word_vocab, result_dir):
    results = {}

    args.test_batch_size = args.per_gpu_eval_batch_size
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.test_batch_size,
                                 collate_fn=my_collate)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.test_batch_size)

    model.eval()
    test_sa_loss, test_sa_accuracy = 0, 0
    test_dt_loss = 0
    nb_test_steps, nb_test_examples = 0, 0
    with open(os.path.join(result_dir, "test_ep_"+str(epoch)+".txt"),"w") as f_test:
        f_test.write('yes_not\tyes_not_pre\tsentence\ttrue_ner\tpredict_ner\n')
        batch_index = 0
        for test_batch in test_dataloader:
            test_batch = tuple(t.to(args.device) for t in test_batch)
            inputs, _ = get_input_from_batch(args, test_batch)

            with torch.no_grad():
                sa_test_loss, dt_test_loss, sa_logits, dt_predict, loss, mask = model(**inputs)

            # category & polarity
            sa_logits = F.softmax(sa_logits, dim=-1)
            sa_logits = sa_logits.detach().cpu().numpy()
            sa_label_ids = inputs["yes_no"].to('cpu').numpy()
            sa_outputs = np.argmax(sa_logits, axis=1)
            dt_label_ids = inputs["target_zo"].to('cpu').numpy()

            sequence_length_line = torch.sum(mask, 1)
            dt_test_tokens = inputs["sentence_cs"]

            for output_i in range(len(sa_outputs)):
                # category & polarity
                f_test.write(str(sa_label_ids[output_i]))
                f_test.write('\t')
                f_test.write(str(sa_outputs[output_i]))
                f_test.write('\t')

                # sentence
                sentence_clean = ['[CLS]']
                dt_label_true = ['[CLS]']
                dt_label_pre = ['[CLS]']
                sentence_len = sequence_length_line[output_i] - 1

                # predict target
                if args.target_method == 'TO':
                    dt_logits = torch.argmax(F.log_softmax(dt_predict[output_i], dim=-1),dim=-1)
                    dt_logits = dt_logits.detach().cpu().numpy()
                    itos_t = {1: 'T', 0: 'O', 2:'<PAD>'}
                else:
                    dt_logits = dt_predict[output_i]
                    itos_t = {1: 'B', 3: 'I', 0: 'O', 2: '<PAD>'}

                for i in range(sentence_len):
                    if not word_vocab['itos'][dt_test_tokens[output_i][i+4].item()].startswith('##'):
                        sentence_clean.append(word_vocab['itos'][dt_test_tokens[output_i][i+4].item()])
                        dt_label_true.append(itos_t[dt_label_ids[output_i][i]])
                        dt_label_pre.append(itos_t[dt_logits[i]])


                f_test.write(' '.join(sentence_clean))
                f_test.write('\t')
                f_test.write(' '.join(dt_label_true))
                f_test.write("\t")
                f_test.write(' '.join(dt_label_pre))
                f_test.write("\n")
            sa_test_accuracy=np.sum(sa_outputs == sa_label_ids)
            test_sa_loss += sa_test_loss.mean().item()
            test_dt_loss += dt_test_loss.mean().item()
            test_sa_accuracy += sa_test_accuracy

            nb_test_examples += inputs["sentence_cs"].size(0)
            nb_test_steps += 1

    test_sa_loss = test_sa_loss / nb_test_steps
    test_dt_loss = test_dt_loss / nb_test_steps
    test_accuracy = test_sa_accuracy / nb_test_examples

    return test_sa_loss, test_dt_loss, test_accuracy

