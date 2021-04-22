import logging
import os
import pickle
import torch
import simplejson as json
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer


logger = logging.getLogger(__name__)

def load_datasets_and_vocabs(args):
    train, test = get_dataset(args.dataset_name, args.target_method)

    # Our model takes unrolled data, currently we don't consider the MAMS cases(future experiments)
    train_reshaped_date = get_reshaped_data(train, args)
    test_reshaped_date = get_reshaped_data(test, args)

    len_token = len(args.tokenizer)
    config = BertConfig.from_pretrained(args.bert_model_dir)
    bert = BertModel.from_pretrained(args.bert_model_dir, config=config)
    bert.resize_token_embeddings(len_token)
    args.bert_model = bert

    logger.info('****** After reshaping ******')
    logger.info('Train set size: %s', len(train_reshaped_date))
    logger.info('Test set size: %s,', len(test_reshaped_date))

    # Build word vocabulary(part of speech, dep_tag) and save pickles.
    word_vocab, dep_tag_vocab, pos_tag_vocab = load_and_cache_vocabs(
        train_reshaped_date+test_reshaped_date, args)


    pos_embedding = torch.nn.init.uniform_(torch.FloatTensor(pos_tag_vocab['len'] - 1, args.posemb_dim), a=-0.15, b=0.15)
    args.pos_embedding = torch.cat((torch.zeros(1, args.posemb_dim), pos_embedding))

    train_dataset = ASBA_Depparsed_Dataset(
        train_reshaped_date, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    test_dataset = ASBA_Depparsed_Dataset(
        test_reshaped_date, args, word_vocab, dep_tag_vocab, pos_tag_vocab)

    return train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab


def read_sentence_depparsed(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data


def get_dataset(dataset_name, target_method):
    rest_15_train = os.path.join("data/experiment_datas", target_method, "cs_train_15_TAS_biaffine_depparsed.json")
    rest_15_test = os.path.join("data/experiment_datas", target_method, "cs_test_15_TAS_biaffine_depparsed.json")

    rest_16_train = os.path.join("data/experiment_datas", target_method, "cs_train_16_TAS_biaffine_depparsed.json")
    rest_16_test = os.path.join("data/experiment_datas", target_method, "cs_test_16_TAS_biaffine_depparsed.json")

    ds_train = {'rest15': rest_15_train,
                'rest16': rest_16_train}
    ds_test = {'rest15': rest_15_test,
               'rest16': rest_16_test}

    train = list(read_sentence_depparsed(ds_train[dataset_name]))
    logger.info('# Read %s Train set: %d', dataset_name, len(train))

    test = list(read_sentence_depparsed(ds_test[dataset_name]))
    logger.info("# Read %s Test set: %d", dataset_name, len(test))
    return train, test

def get_reshaped_data(input_data, args):

    reshaped_date = []

    cls_token = ["[CLS]", ]
    sep_token = ["[SEP]", ]

    # Sentiment counters
    total_counter = defaultdict(int)
    yes_no_lookup = {'0': 0, '1': 1}
    if args.target_method == 'BIO':
        target_lookup = {'B': 1, 'I': 3, 'O': 0}
    else:
        target_lookup = {'T': 1, 'O': 0}
    sentiments_lookup = {'N/A': 0, 'negative': 2, 'positive': 1, 'neutral': 3}

    logger.info('*** Start processing data(reshaping) ***')
    tree_samples = []
    # for seeking 'but' examples
    for e in input_data:
        if args.lower:
            e['tokens'] = [x.lower() for x in e['tokens']]

        # Classify based on POS-tags
        pos_class = e['tags']
        category = e['category']
        sentiment = e['sentiment']
        sentiment_tag = sentiments_lookup[sentiment]
        target_tags = e['target_tags']
        target_zo = []

        for word in category.split(' '):
            if len(args.tokenizer.tokenize(word)) > 1:
                args.tokenizer.add_tokens([word])

        for _ in target_tags:
            target_zo.append(target_lookup[_])
        dependencies = e['dependencies']
        for item in dependencies:
            if item[0] == 'root':
                dependencies.remove(item)
                break

        text_token = []
        for _ in e['tokens']:
            token = args.tokenizer.tokenize(_)
            text_token.extend(token)
        if len(text_token) != len(e['tokens']):
            for i, token in enumerate(text_token):
                if token.startswith('##'):
                    pos_class.insert(i, 'SUBM') # pos_class[i-1]
                    if target_zo[i-1] == 'B':
                        target_zo.insert(i, target_lookup['I'])
                    else:
                        target_zo.insert(i, target_zo[i-1])
                    for item in dependencies:
                        if item[1] > i:
                            item[1] += 1
                        if item[2] > i:
                            item[2] += 1
                    dependencies.append(["wpd", i, i+1])

            assert len(text_token) == len(pos_class)
            assert len(text_token) == len(target_zo)
        else:
            text_token = e['tokens']

        pos_class.insert(0, '<pad>')
        pos_class.insert(0, 'CT')
        pos_class.insert(0, 'CT')
        pos_class.insert(0, '<pad>')
        for item in dependencies:
            item[1] += 1
            item[2] += 1
        for i in range(len(e['tokens'])):
            dependencies.append(["tc", i+2, 0])
            dependencies.append(["tc", i+2, 1])
        dependencies.append(["jh", 1, 0])
        dependencies.append(["jh", 0, 1])
        dep_tag = [i[0] for i in dependencies]

        sentence_cs = cls_token + e['category'].split(' ') + sep_token + text_token + sep_token
        pos_class.append('<pad>')
        input_mask = [1] * len(sentence_cs)
        segment_type_ids = [0] * 4 + [1] * (len(text_token)+1)

        text_len = len(sentence_cs) - 3
        assert len(sentence_cs)  == len(pos_class)

        while len(sentence_cs) < args.max_len:
            sentence_cs.append('[PAD]')
            pos_class.append('<pad>')
            input_mask.append(0)
            segment_type_ids.append(0)

        assert len(sentence_cs) == len(input_mask)
        assert len(sentence_cs) == len(segment_type_ids)
        assert len(sentence_cs) == len(pos_class)

        # reshaping
        reshaped_date.append(
            {'sentence_cs': sentence_cs, 'pos_class': pos_class,
                'sentiment_tag': sentiment_tag, 'target_zo': target_zo, 'dependencies': dependencies, 'dep_tag': dep_tag, 'input_mask': input_mask, 'segment_type_ids': segment_type_ids, 'text_len':text_len})

        total_counter[sentiment] += 1

    logger.info('Total sentiment counter: %s', total_counter)

    return reshaped_date

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    index = 0
    itos = {}
    stoi = {}
    with open(vocab_file, "r") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            itos[index] = token
            stoi[token] = index
            index += 1
    itos[index] = 'style_options'
    stoi['style_options'] = index
    itos[index+1] = 'ambience'
    stoi['ambience'] = index + 1
    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def load_and_cache_vocabs(data, args):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    pkls_path = os.path.join(args.output_dir, 'pkls')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)

    word_vocab = load_vocab(os.path.join(args.bert_model_dir, 'vocab.txt'))

    # Build vocab of dependency tags
    cached_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name,))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        pos_tag_vocab = build_pos_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    return word_vocab, dep_tag_vocab, pos_tag_vocab


def _default_unk_index():
    return 1

def build_text_vocab(data, vocab_size=100000, min_freq=2):
    counter = Counter()
    for d in data:
        s = d['sentence_cs']
        counter.update(s)

    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_pos_tag_vocab(data, vocab_size=1000, min_freq=1):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for d in data:
        tags = d['pos_class']
        counter.update(tags)

    itos = ['<pad>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_dep_tag_vocab(data, vocab_size=1000, min_freq=0):
    counter = Counter()
    for d in data:
        tags = d['dep_tag']
        counter.update(tags)

    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

class ASBA_Depparsed_Dataset(Dataset):
    def __init__(self, data, args, word_vocab, dep_tag_vocab, pos_tag_vocab):
        self.data = data
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        items = e['dep_tag_ids'], e['pos_class'], e['text_len'], e['sentiment_tag'], e['sparse_adj_matrix_dep'], e['sparse_adj_matrix_values'], e['target_zo']
        non_bert_items = (e['sentence_cs_ids'], e['input_mask'], e['segment_type_ids'])
        items_tensor = non_bert_items + items
        items_tensor = tuple(torch.tensor(t) for t in items_tensor)
        return items_tensor

    def generateAdjMatrix(self, dep_edge_list, dep_tag_ids, self_length):
        sparseAdjMatrixDep = [[], [], []]
        sparseAdjMatrixValues = []

        def addedge(type_, from_, to_, value_):
            sparseAdjMatrixDep[0].append(type_)
            sparseAdjMatrixDep[1].append(from_)
            sparseAdjMatrixDep[2].append(to_)
            sparseAdjMatrixValues.append(value_)

        for edge in dep_edge_list:
            dep_tag, fromIndex, toIndex = edge
            if dep_tag == "root" or fromIndex == -1 or toIndex == -1 or fromIndex >= self.args.max_len or toIndex >= self.args.max_len:
                continue
            if dep_tag == "tc":
                addedge(6, fromIndex, toIndex, 1.0)
                addedge(5, toIndex, fromIndex, 1.0)
            elif dep_tag == "dep":
                addedge(1, fromIndex, toIndex, 1.0)
                addedge(2, toIndex, fromIndex, 1.0)
            else:
                addedge(3, fromIndex, toIndex, 1.0)
                addedge(4, toIndex, fromIndex, 1.0)

        self.args.edge_types_num = 7
        for i in range(self_length):
            addedge(0, i, i, 1.0)

        return sparseAdjMatrixDep, sparseAdjMatrixValues

    def convert_features_bert(self, i):
        input_ids = self.args.tokenizer.convert_tokens_to_ids(self.data[i]['sentence_cs'])
        self.data[i]['sentence_cs_ids'] = input_ids

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.data)):
            self.convert_features_bert(i)

            self.data[i]['text_len'] = self.data[i]['text_len']

            self.data[i]['dep_tag_ids'] = [self.dep_tag_vocab['stoi'][w]
                                           for w in self.data[i]['dep_tag']]
            self.data[i]['pos_class'] = [self.pos_tag_vocab['stoi'][w]
                                             for w in self.data[i]['pos_class']]

            self.data[i]['sparse_adj_matrix_dep'], self.data[i]['sparse_adj_matrix_values'] = self.generateAdjMatrix(self.data[i]['dependencies'], self.data[i]['dep_tag_ids'], self.data[i]['text_len'])

def my_collate(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.
    '''
    sentence_cs_ids, input_mask, segment_type_ids, dep_tag_ids, pos_class, text_len, sentiment_tag, sparse_adj_matrix_dep, sparse_adj_matrix_values, target_zo = zip(
        *batch)  # from Dataset.__getitem__()
    text_len = torch.tensor(text_len)
    sentiment_tag = torch.tensor(sentiment_tag)

    # Pad sequences.
    sentence_cs_ids = pad_sequence(
        sentence_cs_ids, batch_first=True, padding_value=0)

    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)
    target_zo = pad_sequence(target_zo, batch_first=True, padding_value=2)

    gcn_et = 7
    seq_len = torch.max(text_len)

    adjmv = torch.stack([torch.sparse.FloatTensor(torch.LongTensor(adjm),
                                                 torch.FloatTensor(adjv),
                                                 torch.Size([gcn_et, seq_len, seq_len])).to_dense() for
                        adjm, adjv in zip(sparse_adj_matrix_dep, sparse_adj_matrix_values)])

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    sentence_cs_ids = sentence_cs_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    sentiment_tag = sentiment_tag[sorted_idx]
    adjmv = adjmv[sorted_idx]
    target_zo = target_zo[sorted_idx]
    input_mask = torch.stack(input_mask)[sorted_idx]
    segment_type_ids = torch.stack(segment_type_ids)[sorted_idx]


    return sentence_cs_ids, pos_class, text_len, adjmv, sentiment_tag, target_zo, input_mask, segment_type_ids,
