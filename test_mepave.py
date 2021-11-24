import os
import argparse

from data_loader_mepave import MEPAVEExtractorDataLoader
from uls_dram.extractors import *
from evaluator import MAEExtractorEvaluator, MEPAVEExtractorEvaluator
from uls_dram.pretrained_encoder_usl_vl_tir import PretrainedEncoderTIR

import random
import numpy as np
import torch
from cfgs.config import config, update_config

from paths import PathConfig
path_cfg = PathConfig()

device_id = 0
torch.cuda.set_device(device_id)

def parse_arguments(model_dir=None):
    parser = argparse.ArgumentParser(description='Argument Parser for ULS-DRAM')

    parser.add_argument("--model_path", dest="model_path", type=str, default=path_cfg.MEPAVE_MODEL_PATH)
    parser.add_argument("--mepave_dataset_dir", dest="mepave_dataset_dir", type=str, default=path_cfg.MEPAVE_PATH)
    parser.add_argument("--mepave_anno_prefix", dest="mepave_anno_prefix", type=str,
                        default='jdair.jave')
    parser.add_argument("--mepave_img_prefix", dest="mepave_img_prefix", type=str,
                        default='product_images')

    parser.add_argument("--batch_size", dest="batch_size", type=int, default=70)
    parser.add_argument("--val_batch_size", dest="val_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=45)
    parser.add_argument("--seed", dest="seed", type=int, default=1234)

    parser.add_argument("--validate_every", dest="validate_every", type=int, default=1)
    parser.add_argument("--validate_every_iter", dest="validate_every_iter", type=int, default=1000)
    parser.add_argument("--print_freq", dest="print_freq", type=int, default=20)
    parser.add_argument("--print_freq_eval", dest="print_freq_eval", type=int, default=20)

    parser.add_argument('--cfg', type=str, help='path to config file', default='cfgs/base_gt_boxes_4x16G.yaml')
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--world_size', type=int, default=8, help='total gpu num')

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)

    return args, config

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_mepave_uls_dram_extractor():
    params, config = parse_arguments()
    print(config)
    print(params)

    del params.local_rank

    if params.seed:
        seed = params.seed
        fix_seed(seed)
    
    print("Constructing data loaders...")
    ex_dl = MEPAVEExtractorDataLoader(params, config)
    ex_ev = MEPAVEExtractorEvaluator(params, ex_dl)
    print("Constructing data loaders...[OK]")

    print('Load pretrain transformer...')
    mytrans = PretrainedEncoderTIR(config, bert_model_name=path_cfg.ROBERTA_SMALL_CHINESE)
    bert_ex = UlsDramMEPAVE(config, params, myvlbert=mytrans, vocab_labels=ex_dl.vocab_labels, tagset_size=len(ex_dl.id2tags.keys()),
        label_maxlen=ex_dl.label_maxlen)
    print('Load pretrain transformer...[OK]')

    if torch.cuda.is_available():
        bert_ex = bert_ex.cuda()

    cur_weight_path = os.path.join(params.model_path)
    print('\nLoading weight file:', cur_weight_path)
    cur_state_dict = torch.load(cur_weight_path)
    bert_ex.load_state_dict(cur_state_dict)

    with torch.no_grad():
        ex_ev.get_accuracy(bert_ex, 'test')

if __name__ == '__main__':
    eval_mepave_uls_dram_extractor()