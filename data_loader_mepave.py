from typing import Dict, List
from util import *
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer
from paths import PathConfig
path_cfg = PathConfig()

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.48, 0.498, 0.531),
                         (0.214, 0.207, 0.207))]
)

tags2id = {'':0,'B':1,'I':2,'O':3}
id2tags = {v:k for k,v in tags2id.items()}

class ExtractorDataset(Dataset):
    def __init__(self, params, tokenizer:BertTokenizer, vocab_labels:List[str], sent_maxlen:int, label_maxlen:int, tags2id_mcls:Dict[str, int],
        x:List[List[str]], img_id:List[str], y:List[str], tag_seqs:List[List[str]]):
        self.params = params
        self.x = x
        self.img_id = img_id
        self.y = y
        self.tag_seqs = tag_seqs
        self.vocab_labels = vocab_labels
        self.eye = np.eye(len(vocab_labels)).astype(int)
        self.sent_maxlen = sent_maxlen
        self.label_maxlen = label_maxlen
        self.tags2id_mcls = tags2id_mcls

        self.num_of_samples = len(x)
        self.tokenizer =  tokenizer
    
    def labels2onehot(self, y):
        idx_list = [ self.vocab_labels.index(attr) for attr in y]
        label = self.eye[idx_list].sum(axis=0).tolist()
        return label
    
    def pad_txt_sequence(self, x):
        max_sequence_length = self.sent_maxlen  + 2

        if type(x) == list: x = "".join(x)
        bert_tokenization = self.tokenizer.tokenize(x)

        if len(bert_tokenization) > max_sequence_length - 2:
            bert_tokenization = bert_tokenization[0: (max_sequence_length - 2)]

        tokens = []
        tokens.append("[CLS]")
        for token in bert_tokenization:
            tokens.append(token)
        tokens.append("[SEP]")

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_sequence_length:
            input_ids.append(0)
        
        return input_ids

    def pad_tag_sequence(self, tag_seq:List[str], attrs:List[str], mask:List[int]):
        max_sequence_length = self.sent_maxlen + 2

        if len(tag_seq) > max_sequence_length - 2:
            tag_seq = tag_seq[0: (max_sequence_length - 2)]

        attrs = [ (self.vocab_labels.index(attr), attr) for attr in attrs]
        attrs = sorted(attrs, key=lambda x: x[0])

        attrs_tag_seqs = []
        for _, attr in attrs:
            cur_tag_seqs = []

            cur_tag_seqs.append(tags2id[''])
            for tag_seq_item in tag_seq:
                if attr in tag_seq_item: 
                    cur_tag_seqs.append(tags2id[tag_seq_item.replace('-'+attr, '')])
                else:
                    cur_tag_seqs.append(tags2id['O'])
            
            while len(cur_tag_seqs) < max_sequence_length:
                cur_tag_seqs.append(tags2id[''])
            
            # cur_tag_seqs = cur_tag_seqs[1:]
            attrs_tag_seqs.append(cur_tag_seqs)

        while len(attrs_tag_seqs) < self.label_maxlen:
            attrs_tag_seqs.append([ tags2id['O'] if valid == 1 and ii != 0 else tags2id[''] for ii, valid in enumerate(mask) ])
        
        return attrs_tag_seqs

    def pad_mcls_tag_sequence(self, tag_seq:List[str]):
        max_sequence_length = self.sent_maxlen + 2
        if len(tag_seq) > max_sequence_length - 2:
            tag_seq = tag_seq[0: (max_sequence_length - 2)]
        
        cur_tag_seq = []
        cur_tag_seq.append(self.tags2id_mcls[''])
        for tag_seq_item in tag_seq:
            cur_tag_seq.append(self.tags2id_mcls[tag_seq_item])

        while len(cur_tag_seq) < max_sequence_length:
            cur_tag_seq.append(self.tags2id_mcls[''])

        # cur_tag_seq = cur_tag_seq[1:]

        return cur_tag_seq

    def pad_ocls_tag_sequence(self, tag_seq:List[str]):
        max_sequence_length = self.sent_maxlen + 2
        if len(tag_seq) > max_sequence_length - 2:
            tag_seq = tag_seq[0: (max_sequence_length - 2)]
        
        cur_tag_seq = []
        cur_tag_seq.append(tags2id[''])
        for tag_seq_item in tag_seq:
            cur_tag_seq.append(tags2id[tag_seq_item.split('-')[0]])

        while len(cur_tag_seq) < max_sequence_length:
            cur_tag_seq.append(tags2id[''])

        # cur_tag_seq = cur_tag_seq[1:]
        return cur_tag_seq

    def gen_mask_sequence(self, x):
        max_sequence_length = self.sent_maxlen  + 2

        mask = []
        mask.extend([1]*(len(x)+1))

        while len(mask) < max_sequence_length:
            mask.append(0)
        
        # mask = mask[1:]
        return mask

    def load_image_transform(self, img_id):
        path = os.path.join(self.params.mepave_dataset_dir, self.params.mepave_img_prefix, img_id.split('_')[-1]+'.jpg')
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = transform(image)
        return image

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        mask = self.gen_mask_sequence(self.x[idx])
        tag_seqs = self.pad_tag_sequence(self.tag_seqs[idx], self.y[idx], mask)
        x = self.pad_txt_sequence(self.x[idx])
        y = self.labels2onehot(self.y[idx])
        mcls_tag_seq = self.pad_mcls_tag_sequence(self.tag_seqs[idx])
        ocls_tag_seq = self.pad_ocls_tag_sequence(self.tag_seqs[idx])
        obj_x = self.load_image_transform(self.img_id[idx])

        obj_x = np.array(obj_x)
        x = np.array(x)
        y = np.array(y)
        tag_seqs = np.array(tag_seqs)
        mask = np.array(mask)
        mcls_tag_seq = np.array(mcls_tag_seq)
        ocls_tag_seq = np.array(ocls_tag_seq)

        return x, obj_x, y, tag_seqs, mask, mcls_tag_seq, ocls_tag_seq

class MEPAVEExtractorDataLoader:
    def __init__(self, params, config):

        self.params = params
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(path_cfg.ROBERTA_SMALL_CHINESE)
        
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        SUBSETS = ('train', 'valid', 'test')

        train_data, valid_data, test_data, self.vocab_labels, self.sent_maxlen, self.label_maxlen, self.label_word_maxlen = self.load_data(SUBSETS)
        self.tags2id_mcls = self.get_tags2id_mcls()
        self.id2tags_mcls = {v:k for k,v in self.tags2id_mcls.items()}

        self.tags2id = tags2id
        self.id2tags = id2tags

        train_dataset = ExtractorDataset(params, self.tokenizer, self.vocab_labels, self.sent_maxlen, self.label_maxlen, self.tags2id_mcls, *train_data)
        self.train_dataloader = DataLoader(train_dataset,
                                                             batch_size=self.params.batch_size,
                                                             shuffle=True, **kwargs)

        valid_dataset = ExtractorDataset(params, self.tokenizer, self.vocab_labels, self.sent_maxlen, self.label_maxlen, self.tags2id_mcls, *valid_data)
        self.valid_dataloader = DataLoader(valid_dataset,
                                                             batch_size=self.params.batch_size,
                                                             shuffle=False, **kwargs)

        test_dataset = ExtractorDataset(params, self.tokenizer, self.vocab_labels, self.sent_maxlen, self.label_maxlen, self.tags2id_mcls, *test_data)
        self.test_dataloader = DataLoader(test_dataset,
                                                             batch_size=self.params.batch_size,
                                                             shuffle=False, **kwargs)

    def load_data(self, subsets:List[str]):
        print('calculating vocabulary...')
        ret = []
        sent_maxlen = 0
        label_maxlen = 0
        label_word_maxlen = 0
        vocab_labels = set()

        for subset in subsets:
            sentences, img_id, labels, tag_seqs, cur_vocab_labels, cur_sent_maxlen, cur_label_maxlen, cur_label_word_maxlen = self.load_subset(subset)
            ret.append((sentences, img_id, labels, tag_seqs))

            vocab_labels |= cur_vocab_labels
            if cur_sent_maxlen > sent_maxlen: sent_maxlen = cur_sent_maxlen
            if cur_label_maxlen > label_maxlen: label_maxlen = cur_label_maxlen
            if cur_label_word_maxlen > label_word_maxlen: label_word_maxlen = cur_label_word_maxlen
        
        vocab_labels = sorted(list(vocab_labels))
        ret.extend([vocab_labels, sent_maxlen, label_maxlen, label_word_maxlen])
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
        
        return tags2id_mcls

    def load_subset(self, subset):
        
        fin = os.path.join(self.params.mepave_dataset_dir, '%s.%s.txt' % (self.params.mepave_anno_prefix, subset))
        lines = open(fin, "r", encoding="utf-8").read().strip().split("\n")
        input_seqs = []
        output_seqs = []
        output_labels = []
        img_cids = []
        vocab_labels = set()

        sent_maxlen = 0
        label_maxlen = 0
        label_word_maxlen = 0
        
        for line in lines:
            items = line.split("\t")
            cid = items[0]
            doc = items[2].lower()
            doc_p = items[3].lower()

            input_seq = []
            output_seq = []
            attrs = []
            index = 0

            try:
                assert " " not in doc + doc_p
                while index < len(doc_p):
                    if doc_p[index] == "<":
                        index += 1
                        attr = ""
                        while doc_p[index] != ">":
                            attr += doc_p[index]
                            index += 1
                        index += 1
                        input_seq.append(doc_p[index])
                        output_seq.append("B-"+attr)
                        index += 1
                        while doc_p[index] != "<":
                            input_seq.append(doc_p[index])
                            output_seq.append("I-"+attr)
                            index += 1
                        index += 1
                        assert doc_p[index] == "/"
                        index += 1
                        attr_end = ""
                        while doc_p[index] != ">":
                            attr_end += doc_p[index]
                            index += 1
                        index += 1
                        assert attr_end == attr
                        attrs.append(attr)
                    else:
                        input_seq.append(doc_p[index])
                        output_seq.append("O")
                        index += 1
                assert "".join(input_seq) == doc
                
                img_cids.append(cid)
                input_seqs.append(input_seq)
                output_seqs.append(output_seq)
                if len(output_seq) > sent_maxlen:
                    sent_maxlen = len(output_seq)

                if attrs == []:
                    attrs = ["[PAD]"]
                output_labels.append(sorted(list(set(attrs))))
                vocab_labels |= set(attrs)
                if len(set(attrs)) > label_maxlen:
                    label_maxlen = len(set(attrs))
            
            except (AssertionError, IndexError):
                print("wrong line:", doc, doc_p, "".join(input_seq), "".join(output_seq))
                #exit()

        assert len(input_seqs) == len(output_seqs) == len(output_labels) == len(img_cids)

        print('subset ', subset)
        print('sent_maxlen', sent_maxlen)
        print('number sentence', len(output_seqs))
        print('attr_label_maxlen', label_maxlen)
        print('number image', len(img_cids))

        print('='*32)

        label_word_maxlen = max([ len(label) for label in vocab_labels ])

        return [input_seqs, img_cids, output_labels, output_seqs, vocab_labels, sent_maxlen, label_maxlen, label_word_maxlen]

if __name__ == '__main__':
    pass