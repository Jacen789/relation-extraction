import re
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))


class MyTokenizer(object):
    def __init__(self, pretrained_model_path=None, mask_entity=False):
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.mask_entity = mask_entity

    def tokenize(self, item):
        sentence = item['text']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            pos_min = pos_head
            pos_max = pos_tail
            rev = False

        sent0 = self.bert_tokenizer.tokenize(sentence[:pos_min[0]])
        ent0 = self.bert_tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        sent1 = self.bert_tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = self.bert_tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = self.bert_tokenizer.tokenize(sentence[pos_max[1]:])

        if rev:
            if self.mask_entity:
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_tail = [len(sent0), len(sent0) + len(ent0)]
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        else:
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        tokens = sent0 + ent0 + sent1 + ent1 + sent2

        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = [0, 0]
        pos2 = [0, 0]
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1[0] = len(re_tokens)
                re_tokens.append('[unused1]')
            if cur_pos == pos_tail[0]:
                pos2[0] = len(re_tokens)
                re_tokens.append('[unused2]')
            re_tokens.append(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused3]')
                pos1[1] = len(re_tokens)
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused4]')
                pos2[1] = len(re_tokens)
            cur_pos += 1
        re_tokens.append('[SEP]')
        return re_tokens[1:-1], pos1, pos2


def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask


def read_data(input_file, tokenizer=None, max_len=128):
    tokens_list = []
    e1_mask_list = []
    e2_mask_list = []
    tags = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in):
            line = line.strip()
            item = json.loads(line)
            if tokenizer is None:
                tokenizer = MyTokenizer()
            tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
            if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and \
                    pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
                tokens_list.append(tokens)
                e1_mask = convert_pos_to_mask(pos_e1, max_len)
                e2_mask = convert_pos_to_mask(pos_e2, max_len)
                e1_mask_list.append(e1_mask)
                e2_mask_list.append(e2_mask)
                tag = item['relation']
                tags.append(tag)
    return tokens_list, e1_mask_list, e2_mask_list, tags


def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))


def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


class SentenceREDataset(Dataset):
    def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=128):
        self.data_file_path = data_file_path
        self.tagset_path = tagset_path
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)
        self.max_len = max_len
        self.tokens_list, self.e1_mask_list, self.e2_mask_list, self.tags = read_data(data_file_path, tokenizer=self.tokenizer, max_len=self.max_len)
        self.tag2idx = get_tag2idx(self.tagset_path)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        sample_e1_mask = self.e1_mask_list[idx]
        sample_e2_mask = self.e2_mask_list[idx]
        sample_tag = self.tags[idx]
        encoded = self.tokenizer.bert_tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']
        sample_tag_id = self.tag2idx[sample_tag]

        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'e1_mask': torch.tensor(sample_e1_mask),
            'e2_mask': torch.tensor(sample_e2_mask),
            'tag_id': torch.tensor(sample_tag_id)
        }
        return sample
