import os
import json
import pickle
import pandas as pd
from transformers import BertTokenizer
from paths import PathConfig

from tqdm import tqdm
tqdm.pandas()

path_cfg = PathConfig()

BERT_PATH = path_cfg.BERT_BASE_UNCASED
DATA_PATH = path_cfg.MAE_PATH

IMG_PATH = 'mumie-img'
SAVE_PREFIX = 'mae.'
SUBSETS = ('train', 'val')

# maximal text sequence length
MAX_LEN = 128
def convert_file_format(subsets):
    for subset in subsets:
        print('converting subset:', subset)
        tokenizer:BertTokenizer = BertTokenizer.from_pretrained(BERT_PATH)

        file_list = sorted(os.listdir(os.path.join(DATA_PATH, subset)))
        item_list, img_list, tokens_list, tag_seq_list = [], [], [], []

        for file_idx, file_name in enumerate(file_list):
            ann_list = json.load(open(os.path.join(DATA_PATH, subset, file_name)))
            print('processing file:', file_name, str(file_idx+1)+'/'+str(len(file_list)))

            total_item_count, valid_item_count = 0, 0
            for item_idx, ann_item in enumerate(ann_list):
                img_full_path = os.path.join(DATA_PATH, IMG_PATH, ann_item['images'][0])
                total_item_count += 1
                if not os.path.exists(img_full_path): continue
                else: valid_item_count += 1

                title = ann_item['title']
                text = ann_item['text']
                text = title + ' ' + text
                tokens = tokenizer.tokenize(text)[:MAX_LEN-2]
                tag_seq = ['O'] * len(tokens)

                tokens.insert(0, '[CLS]')
                tokens.append('[SEP]')
                while len(tokens) < MAX_LEN:
                    tokens.append('[PAD]')
                tag_seq.insert(0, '')
                while len(tag_seq) < MAX_LEN:
                    tag_seq.append('')

                attrvals = ann_item['specs']

                for attr in attrvals:
                    value = attrvals[attr]
                    value_tokens = tokenizer.tokenize(value)
                    # print(attr, value_tokens)
                    for idx in range(len(tokens)):

                        if ' '.join(tokens[idx:idx+len(value_tokens)]) == ' '.join(value_tokens):
                            for idx_to_tag, token_to_tag in enumerate(tokens[idx:idx+len(value_tokens)]):
                                if idx_to_tag == 0: tag_seq[idx+idx_to_tag] = 'B-'+attr
                                elif '##' in token_to_tag: tag_seq[idx+idx_to_tag] = 'X-'+attr
                                else: tag_seq[idx+idx_to_tag] = 'I-'+attr
                
                item_list.append(' '.join([file_name, str(item_idx)]))
                img_list.append(ann_item['images'][0])

                assert len(tokens) == MAX_LEN and len(tag_seq) == MAX_LEN
                tokens_list.append(tokens)
                tag_seq_list.append(tag_seq)

            print('total item count:', total_item_count)
            print('valid item count:', valid_item_count, '\n')
        
        data = pd.DataFrame({ 'item_id': item_list, 'img_file': img_list, 'txt_tokens': tokens_list, 'tag_seq': tag_seq_list })
        pickle.dump(data, open(os.path.join(DATA_PATH, SAVE_PREFIX+subset+'.pkl'), 'wb'))

def get_vocab_labels(subset='train'):
    pkl_file = open(os.path.join(DATA_PATH, SAVE_PREFIX+subset+'.pkl'), 'rb')
    data : pd.DataFrame = pickle.load(pkl_file)
    vocab_per_item = data[['tag_seq']].progress_apply(lambda x: [ item.split('B-')[1:][0] for item in x[0] if 'B-' in item], axis=1)

    vocab_list = set()
    for vocab_item in tqdm(vocab_per_item.tolist()):
        vocab_list |= set(vocab_item)
    vocab_list = list(vocab_list)

    print('attribute vocab count:', len(vocab_list))
    vocab_file = open(os.path.join(DATA_PATH, SAVE_PREFIX+'vocab.txt'), 'w')
    vocab_file.writelines([ vocab+'\n' for vocab in vocab_list])

if __name__ == '__main__':
    convert_file_format(SUBSETS)
    get_vocab_labels()
    pass
