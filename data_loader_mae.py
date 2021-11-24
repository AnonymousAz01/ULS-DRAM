import os
import json
import pickle
import torch
import traceback
import pandas as pd
import numpy as np

import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, List
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
tqdm.pandas()
from transformers import BertTokenizer
import random
from paths import PathConfig

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.48, 0.498, 0.531),
                         (0.214, 0.207, 0.207))]
)

tags2id = {'':0,'B':1,'I':2,'O':3, 'X':4}
id2tags = {v:k for k,v in tags2id.items()}

path_cfg = PathConfig()

BERT_PATH = path_cfg.BERT_BASE_UNCASED
DATA_PATH = path_cfg.MAE_PATH

IMG_PATH = 'mumie-img'
SAVE_PREFIX = 'mae.'
SUBSETS = ('train', 'val')
# maximal text sequence length
MAX_LEN = 128

class MAEExtractorDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, vocab_labels:List[str], sent_maxlen:int, label_maxlen:int, tags2id_mcls:Dict[str, int], 
        img_id:List[str], x:List[List[str]], tag_seqs:List[List[str]], name:str=''):
        self.x = x
        self.img_id = img_id
        self.tag_seqs = tag_seqs
        self.vocab_labels = vocab_labels
        self.eye = np.eye(len(vocab_labels)).astype(int)
        self.sent_maxlen = sent_maxlen
        self.label_maxlen = label_maxlen
        self.tags2id_mcls = tags2id_mcls

        self.num_of_samples = len(x)
        self.tokenizer =  tokenizer
        self.name = name
        pass
    
    def __len__(self):
        return self.num_of_samples
    
    def __getitem__(self, idx):
        mask = self.gen_mask_sequence(self.x[idx])

    def gen_mask_sequence(self, x):
        mask = []
        for item in x:
            if item not in ('[PAD]', ):
                mask.append(1)
            else: mask.append(0)
        
        return mask
    
    def gen_txt_sequence(self, x):
        return self.tokenizer.convert_tokens_to_ids(x)
    
    def gen_attr_list(self, tag_seqs):
        return set([ item.split('B-')[1:][0] for item in tag_seqs if ('B-' in item) and (item.split('B-')[1:][0] in self.vocab_labels)])
    
    def gen_tag_sequence(self, tag_seq:List[str], attrs:List[str], mask:List[int]):
        attrs = [ (self.vocab_labels.index(attr), attr) for attr in attrs]
        attrs = sorted(attrs, key=lambda x: x[0])
        attrs_tag_seqs = []
        for _, attr in attrs:
            cur_tag_seqs = []

            for tag_seq_item in tag_seq:
                if any([tag_prefix+'-'+attr == tag_seq_item for tag_prefix in ('B', 'I', 'X')]) : 
                    if 'X' in tag_seq_item: cur_tag_seqs.append(tags2id['X'])
                    else: cur_tag_seqs.append(tags2id[tag_seq_item.replace('-'+attr, '')])
                elif any([tag_prefix in tag_seq_item for tag_prefix in ('B', 'I', 'X')]):
                    cur_tag_seqs.append(tags2id['O'])
                else:
                    cur_tag_seqs.append(tags2id[tag_seq_item])
            
            attrs_tag_seqs.append(cur_tag_seqs)
        
        while len(attrs_tag_seqs) < self.label_maxlen:
            attrs_tag_seqs.append([ tags2id['O'] if valid == 1 and ii != 0 else tags2id[''] for ii, valid in enumerate(mask) ])

        return attrs_tag_seqs
    
    def gen_mcls_tag_sequence(self, tag_seq:List[str]):
        cur_tag_seq = []
        for tag_seq_item in tag_seq:
            if 'X' in tag_seq_item:
                cur_tag_seq.append(self.tags2id_mcls['X'])
            elif any([tag_seq_item.split(tag_prefix)[0] not in self.vocab_labels for tag_prefix in ('B', 'I', 'X')]):
                cur_tag_seq.append(self.tags2id_mcls['O'])
            else:
                cur_tag_seq.append(self.tags2id_mcls[tag_seq_item])
        return cur_tag_seq

    def gen_ocls_tag_sequence(self, tag_seq:List[str]):
        return [ tags2id[tag_seq_item.split('-')[0]] for tag_seq_item in tag_seq ]

    def load_image_transform(self, img_id):
        path = os.path.join(DATA_PATH, IMG_PATH, img_id)
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = transform(image)
        return image
    
    def labels2onehot(self, y):
        idx_list = [ self.vocab_labels.index(attr) for attr in y]
        label = self.eye[idx_list].sum(axis=0).tolist()
        return label

    def __getitem__(self, idx):
        try:
            mask = self.gen_mask_sequence(self.x[idx])
            y = self.gen_attr_list(self.tag_seqs[idx])
            tag_seqs = self.gen_tag_sequence(self.tag_seqs[idx], y, mask)
            x = self.gen_txt_sequence(self.x[idx])
            y = self.labels2onehot(y)

            mcls_tag_seq = self.gen_mcls_tag_sequence(self.tag_seqs[idx])
            ocls_tag_seq = self.gen_ocls_tag_sequence(self.tag_seqs[idx])

            obj_x = self.load_image_transform(self.img_id[idx])
            obj_x = np.array(obj_x)

            x = np.array(x)
            y = np.array(y)
            tag_seqs = np.array(tag_seqs)
            mask = np.array(mask)
            mcls_tag_seq = np.array(mcls_tag_seq)
            ocls_tag_seq = np.array(ocls_tag_seq)

            return x, obj_x, y, tag_seqs, mask, mcls_tag_seq, ocls_tag_seq
        except:
            traceback.print_exc()
            print('%s dataset image file:' % self.name, self.img_id[idx])
            return self.__getitem__(int(random.random()*self.num_of_samples-1))

class MAEExtractorDataLoader:
    def __init__(self, params, config):
        self.params = params
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(path_cfg.BERT_BASE_UNCASED)
        self.sent_maxlen = MAX_LEN
        
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_data, val_data, self.vocab_labels, self.label_maxlen = self.load_data(SUBSETS)
        self.tags2id_mcls = self.get_tags2id_mcls()
        self.id2tags_mcls = {v:k for k,v in self.tags2id_mcls.items()}

        self.tags2id = tags2id
        self.id2tags = id2tags

        train_dataset = MAEExtractorDataset(self.tokenizer, self.vocab_labels, self.sent_maxlen, self.label_maxlen, self.tags2id_mcls, *train_data, name='train')
        if 'local_rank' in self.params:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            self.train_dataloader = DataLoader(train_dataset,
                                                                batch_size=self.params.batch_size,
                                                                #  shuffle=True, 
                                                                sampler=train_sampler, **kwargs)
        else:
            self.train_dataloader = DataLoader(train_dataset,
                                                                batch_size=self.params.batch_size,
                                                                 shuffle=True, **kwargs)

        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        val_dataset = MAEExtractorDataset(self.tokenizer, self.vocab_labels, self.sent_maxlen, self.label_maxlen, self.tags2id_mcls, *val_data, name='val')
        self.valid_dataloader = DataLoader(val_dataset,
                                                             batch_size=self.params.val_batch_size,
                                                             shuffle=False, **kwargs)

    def load_data(self, subsets:List[str]):
        ret = []
        label_maxlen, cur_label_maxlen = 0, 0
        for subset in subsets:
            pkl_file = open(os.path.join(DATA_PATH, SAVE_PREFIX+subset+'.pkl'), 'rb')
            data:pd.DataFrame = pickle.load(pkl_file)
            ret.append(( data['img_file'].tolist(), data['txt_tokens'].tolist(), data['tag_seq'].tolist(), ))
            cur_label_maxlen = max([ len(set([ item.split('B-')[1:][0] for item in tag_seqs if 'B-' in item])) for tag_seqs in data['tag_seq'].tolist() ])
            label_maxlen = max([label_maxlen, cur_label_maxlen])
        
        vocab_file = open(os.path.join(DATA_PATH, SAVE_PREFIX+'vocab.txt'), 'r')
        vocab_labels = [ line.strip('\n') for line in vocab_file.readlines() ]

        ret.extend([ vocab_labels, label_maxlen])

        return ret

    def get_tags2id_mcls(self):
        tags2id_mcls = {'':0}
        idx = 1
        for attr in self.vocab_labels:
            tags2id_mcls['B-'+attr] = idx
            idx += 1
            tags2id_mcls['I-'+attr] = idx
            idx += 1
        
        tags2id_mcls['O'] = idx
        idx += 1
        tags2id_mcls['X'] = idx
        
        return tags2id_mcls
