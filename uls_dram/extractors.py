from typing import Dict, List
from pytorch_pretrained_bert.modeling import BertModel
from uls_dram.pretrained_encoder_usl_vl import PretrainedEncoderUlsVL
from uls_dram.pretrained_encoder_usl_vl_tir import PretrainedEncoderTIR
import torch.nn as nn
from uls_dram.util import *
from torchcrf import CRF
import data_loader_mae as ex_dl_mae
from data_loader_mepave import id2tags
from paths import PathConfig

path_cfg = PathConfig()

# ULS-DRAM on MAE 
# (integrated loss function for DDP Training on multiple GPUs)
class UlsDramMAE(nn.Module):
    def __init__(self, config, params, myvlbert:PretrainedEncoderTIR, vocab_labels:List[str], label_maxlen=2, tagset_size=4, cls_thr=0.4):
        super(UlsDramMAE, self).__init__()
        self.config = config
        self.params = params

        self.label_maxlen = label_maxlen
        self.hidden_size = self.config.NETWORK.VLBERT.hidden_size
        self.attn_size = 200
        self.device = 'cuda'

        self.tagset_size = tagset_size
        self.dropout_rate = self.config.NETWORK.CLASSIFIER_DROPOUT

        self.label_embeddings = nn.Embedding(len(vocab_labels), self.hidden_size)

        self.mybert:PretrainedEncoderTIR = myvlbert
        self.attr_cls = nn.Sequential(
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=len(vocab_labels))
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.cls_thr = cls_thr

        #CRF
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_size * 2, self.tagset_size)
        self.sigmoid = nn.Sigmoid()
        self.loss_function_crf = CRF(self.tagset_size, batch_first=True)
        self.loss_function_cls = nn.BCEWithLogitsLoss()
    
    def calculate_cosin(self, context_output, att_hidden):
        '''
        context_output (batchsize, seqlen, hidden_dim)
        att_hidden (batchsize, hidden_dim)
        '''
        batch_size,seqlen,hidden_dim = context_output.size()
        att_hidden = att_hidden.unsqueeze(1).repeat(1,seqlen,1)

        context_output = context_output.float()
        att_hidden = att_hidden.float()

        cos = torch.sum(context_output*att_hidden, dim=-1)/(torch.norm(context_output, dim=-1)*torch.norm(att_hidden, dim=-1))
        cos = cos.unsqueeze(-1)
        cos_output = context_output*cos
        outputs = torch.cat([context_output, cos_output], dim=-1)

        return outputs

    def tag_postprocess(self, ground_seqs, ground_scores, pred_batch_idxs, pred_word_idxs, mask):
        batch_size = ground_scores.shape[0]
        
        ground_indexs = [ [] for _ in range(batch_size) ]
        ground_indexs_map = np.nonzero(ground_scores.cpu().numpy())
        for idx, value in zip(*ground_indexs_map):
            ground_indexs[idx] += [value]

        pred_indexs = [ [] for _ in range(batch_size) ]
        for idx, value in zip(pred_batch_idxs, pred_word_idxs):
            pred_indexs[idx] += [value]

        new_ground_seqs = []
        for batch_idx, (cur_ground_indexs, cur_pred_indexs) in enumerate(zip(ground_indexs, pred_indexs)):
            for cur_pred_index in cur_pred_indexs:
                if cur_pred_index in cur_ground_indexs:
                    new_ground_seqs.append(ground_seqs[batch_idx][ground_indexs[batch_idx].index(cur_pred_index)].tolist())
                else:
                    new_ground_seqs.append([ ex_dl_mae.tags2id['O'] if valid and ii != 0 else ex_dl_mae.tags2id[''] for ii, valid in enumerate(mask[batch_idx].bool()) ])

        mask = mask[pred_batch_idxs]
        
        return torch.Tensor(new_ground_seqs), mask

    def loss_function(self, 
        batch_idxs, word_idxs, cls_scores, seq_outputs,
        tag_seqs, y, mask, 
        ):
        loss_cls = self.loss_function_cls(cls_scores, y.float())

        new_tag_seqs, crf_mask = self.tag_postprocess(tag_seqs, y, batch_idxs, word_idxs, mask)

        new_tag_seqs = new_tag_seqs.long().cuda()
        seq_outputs = seq_outputs.cuda()
        crf_mask = crf_mask.bool().cuda()

        loss_tag = -self.loss_function_crf(seq_outputs, new_tag_seqs, crf_mask)

        return loss_cls, loss_tag

    def forward(self, sentence, img_obj, 
        cls_train_flag=False,
        tag_seqs=None, y=None, mask=None
        ):

        batch_size = sentence.shape[0]

        all_encoder_layers, attention_probs = self.mybert(img_obj, sentence)
        cls_token = torch.cat([
            all_encoder_layers[-2][:, :1, :], all_encoder_layers[-1][:, :1, :], 
        ], dim=-1)

        cls_scores = self.attr_cls(cls_token)
        cls_scores = cls_scores.view(cls_scores.shape[0], cls_scores.shape[-1])
        cls_scores_active = self.sigmoid(cls_scores)
        
        batch_idxs, word_idxs = torch.where(cls_scores_active > self.cls_thr)

        # no class predicted from cls stage
        if len(batch_idxs) == 0 or cls_train_flag: 
            if tag_seqs != None:
                # training
                return self.loss_function_cls(cls_scores, y.float()), torch.Tensor([0]).cuda()
            else:
                # evaluation
                return cls_scores, batch_idxs, word_idxs, torch.Tensor([])
        
        # some class predicted from cls stage, perform tag stage
        else:
            context_output = all_encoder_layers[-1]
            context_output = context_output[batch_idxs]
            attr_hidden = self.label_embeddings(word_idxs)
            
            outputs = self.calculate_cosin(context_output, attr_hidden)

            outputs, _ = self.lstm(outputs)
            outputs = self.dropout(outputs)
            outputs = self.hidden2tag(outputs)

            if tag_seqs != None:
                # training
                return self.loss_function(
                    batch_idxs, word_idxs, cls_scores, outputs,
                    tag_seqs, y, mask
                )
            else:
                # evalutation
                return cls_scores, batch_idxs, word_idxs, outputs

# ULS-DRAM on MEPAVE
class UlsDramMEPAVE(nn.Module):
    def __init__(self, config, params, myvlbert:PretrainedEncoderTIR, vocab_labels:List[str], label_maxlen=2, tagset_size=4, label_word_maxlen=10):
        super(UlsDramMEPAVE, self).__init__()
        self.config = config
        self.params = params

        self.label_maxlen = label_maxlen
        self.hidden_size = self.config.NETWORK.VLBERT.hidden_size
        self.attn_size = 200
        self.device = 'cuda'

        self.tagset_size = tagset_size
        self.dropout_rate = self.config.NETWORK.CLASSIFIER_DROPOUT

        self.label_embeddings = nn.Embedding(len(vocab_labels), self.hidden_size)
        self.mybert:PretrainedEncoderTIR = myvlbert

        self.attr_cls = nn.Sequential(
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=len(vocab_labels))
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.cls_thr = 0.5

        #CRF
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_size * 2, self.tagset_size)
        self.sigmoid = nn.Sigmoid()
        self.loss_function_crf = CRF(len(id2tags.keys()), batch_first=True)
    
    def calculate_cosin(self, context_output, att_hidden):
        '''
        context_output (batchsize, seqlen, hidden_dim)
        att_hidden (batchsize, hidden_dim)
        '''
        batch_size,seqlen,hidden_dim = context_output.size()
        att_hidden = att_hidden.unsqueeze(1).repeat(1,seqlen,1)

        context_output = context_output.float()
        att_hidden = att_hidden.float()

        cos = torch.sum(context_output*att_hidden, dim=-1)/(torch.norm(context_output, dim=-1)*torch.norm(att_hidden, dim=-1))
        cos = cos.unsqueeze(-1)
        cos_output = context_output*cos
        outputs = torch.cat([context_output, cos_output], dim=-1)

        return outputs

    def forward(self, sentence, img_obj, visualize=False):
        batch_size = sentence.shape[0]

        all_encoder_layers, attention_probs = self.mybert(img_obj, sentence)
        cls_token = torch.cat([
            all_encoder_layers[-2][:, :1, :], all_encoder_layers[-1][:, :1, :], 
        ], dim=-1)

        cls_scores = self.attr_cls(cls_token)
        cls_scores = cls_scores.view(cls_scores.shape[0], cls_scores.shape[-1])
        cls_scores_active = self.sigmoid(cls_scores)
        
        batch_idxs, word_idxs = torch.where(cls_scores_active > self.cls_thr)

        # no class predicted from cls stage
        if len(batch_idxs) == 0: return cls_scores, batch_idxs, word_idxs, torch.Tensor([])

        context_output = all_encoder_layers[-1]
        context_output = context_output[batch_idxs]
        attr_hidden = self.label_embeddings(word_idxs)
        
        outputs = self.calculate_cosin(context_output, attr_hidden)

        outputs, _ = self.lstm(outputs)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        
        if visualize:
            return cls_scores, batch_idxs, word_idxs, outputs, attention_probs
        else:
            return cls_scores, batch_idxs, word_idxs, outputs

# Unified Learning Scheme for Language Only
# Only ULS-L with a pretrained jointly finetune BERT
class UnifiedLearningSchemeLanguageOnly(nn.Module):
    def __init__(self, config, params, tags2id_mcls:Dict[str, int], ):
        super(UnifiedLearningSchemeLanguageOnly, self).__init__()
        self.config = config
        self.params = params

        self.bert:BertModel = BertModel.from_pretrained(path_cfg.ROBERTA_SMALL_CHINESE)
        self.dropout = nn.Dropout(self.config.NETWORK.CLASSIFIER_DROPOUT)

        #CRF
        self.lstm = nn.LSTM(self.config.NETWORK.VLBERT.hidden_size, self.config.NETWORK.VLBERT.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.config.NETWORK.VLBERT.hidden_size, len(tags2id_mcls.keys()))
        self.loss_function_crf = CRF(len(tags2id_mcls.keys()), batch_first=True)
    
    def forward(self, sentence):
        all_encoder_layers, _ = self.bert(sentence)
        context_output, _ = self.lstm(all_encoder_layers[-1])
        outputs = self.dropout(context_output)

        outputs = self.hidden2tag(outputs)

        return outputs

# Unified Learning Scheme for Vision and Language
# ULS-VL with a pretrained jointly finetune BERT and ResNet (From PretrainedEncoderUlsVL)
class UnifiedLearningSchemeVisualLanguage(nn.Module):
    def __init__(self, config, params, myvlbert:PretrainedEncoderUlsVL, tags2id_mcls:Dict[str, int], ):
        super(UnifiedLearningSchemeVisualLanguage, self).__init__()
        self.config = config
        self.params = params
        self.mybert:PretrainedEncoderUlsVL = myvlbert
        self.dropout = nn.Dropout(self.config.NETWORK.CLASSIFIER_DROPOUT)

        #CRF
        self.lstm = nn.LSTM(self.config.NETWORK.VLBERT.hidden_size, self.config.NETWORK.VLBERT.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.config.NETWORK.VLBERT.hidden_size, len(tags2id_mcls.keys()))
        self.loss_function_crf = CRF(len(tags2id_mcls.keys()), batch_first=True)
    
    def forward(self, sentence, img_obj, visualize=False):
        all_encoder_layers, attention_probs = self.mybert(img_obj, sentence)
        context_output, _ = self.lstm(all_encoder_layers[-1])
        outputs = self.dropout(context_output)

        outputs = self.hidden2tag(outputs)

        if not visualize:
            return outputs
        else:
            return outputs, attention_probs

# DyNet Guided Policy based on ULS-VL+TIR (PretrainedEncoderTIR)
class DynGuidedExtractor(nn.Module):
    def __init__(self, config, params, myvlbert:PretrainedEncoderTIR, vocab_labels:List[str], label_maxlen=2, tagset_size=4, label_word_maxlen=10):
        super(DynGuidedExtractor, self).__init__()
        self.config = config
        self.params = params

        self.label_maxlen = label_maxlen
        self.hidden_size = self.config.NETWORK.VLBERT.hidden_size
        self.tagset_size = tagset_size

        self.label_embeddings = nn.Embedding(len(vocab_labels), self.hidden_size * self.hidden_size)

        self.mybert:PretrainedEncoderTIR = myvlbert

        self.attr_cls = nn.Sequential(
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=len(vocab_labels))
        )
        self.dropout = nn.Dropout(self.config.NETWORK.CLASSIFIER_DROPOUT)

        #CRF
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        
        self.hidden2tag = nn.Linear(self.hidden_size * 2, self.tagset_size)
        self.sigmoid = nn.Sigmoid()
        self.loss_function_crf = CRF(len(id2tags.keys()), batch_first=True)
    
    def forward(self, sentence, img_obj):
        batch_size = sentence.shape[0]

        all_encoder_layers, attention_probs = self.mybert(img_obj, sentence)
        cls_token = torch.cat([
            # all_encoder_layers[-3][:, :1, :], 
            all_encoder_layers[-2][:, :1, :], all_encoder_layers[-1][:, :1, :], 
        ], dim=-1)

        cls_scores = self.attr_cls(cls_token)
        cls_scores = cls_scores.view(cls_scores.shape[0], cls_scores.shape[-1])
        cls_scores_active = self.sigmoid(cls_scores)
        
        batch_idxs, word_idxs = torch.where(cls_scores_active > 0.5)
        # no class predicted from cls stage
        if len(batch_idxs) == 0: return cls_scores, batch_idxs, word_idxs, torch.Tensor([])

        context_output = all_encoder_layers[-1]
        context_output = context_output[batch_idxs]
        attr_hidden = self.label_embeddings(word_idxs)
        attr_hidden = attr_hidden.view(-1, self.hidden_size, self.hidden_size)
        
        attr_outputs = torch.matmul(context_output, attr_hidden)
        outputs = torch.cat([context_output, attr_outputs], dim=-1)

        outputs, _ = self.lstm(outputs)

        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)

        return cls_scores, batch_idxs, word_idxs, outputs

# Bert Guided Policy based on ULS-VL+TIR (PretrainedEncoderTIR)
class BertGuidedExtractor(nn.Module):
    def __init__(self, config, params, myvlbert:PretrainedEncoderTIR, vocab_labels:List[str], label_maxlen=2, tagset_size=4, label_word_maxlen=10):
        super(BertGuidedExtractor, self).__init__()
        self.config = config
        self.params = params
        self.tokenizer = PretrainedEncoderTIR.from_pretrained(path_cfg.ROBERTA_SMALL_CHINESE)
        vocab_labels = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label)) for label in vocab_labels]

        self.pad_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])
        self.cls_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        self.sep_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])

        vocab_labels = [ self.cls_id+label+self.sep_id+self.pad_id*(label_word_maxlen - len(label)) for label in vocab_labels ]
        self.label_maxlen = label_maxlen

        self.vocab_labels = torch.Tensor(vocab_labels)

        self.hidden_size = self.config.NETWORK.VLBERT.hidden_size

        self.mybert:PretrainedEncoderTIR = myvlbert
        # self.attr_cls = nn.Linear(in_features=self.config.NETWORK.VLBERT.hidden_size, out_features=len(vocab_labels))
        self.attr_cls = nn.Sequential(
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=len(vocab_labels))
        )
        self.bert:BertModel = BertModel.from_pretrained(self.config.NETWORK.BERT_MODEL_NAME)
        self.dropout = nn.Dropout(self.config.NETWORK.CLASSIFIER_DROPOUT)

        #CRF
        self.lstm = nn.LSTM(self.config.NETWORK.VLBERT.hidden_size * 2, self.config.NETWORK.VLBERT.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.config.NETWORK.VLBERT.hidden_size * 2, tagset_size)
        self.sigmoid = nn.Sigmoid()
        self.loss_function_crf = CRF(len(id2tags.keys()), batch_first=True)

        self.cls_encoder_layer_idx = self.params.cls_encoder_layer_idx

    def calculate_cosin(self, context_output, att_hidden):
        '''
        context_output (batchsize, seqlen, hidden_dim)
        att_hidden (batchsize, hidden_dim)
        '''
        batchsize,seqlen,hidden_dim = context_output.size()
        att_hidden = att_hidden.repeat(1,seqlen,1)

        context_output = context_output.float()
        att_hidden = att_hidden.float()

        cos = torch.sum(context_output*att_hidden, dim=-1)/(torch.norm(context_output, dim=-1)*torch.norm(att_hidden, dim=-1))
        cos = cos.unsqueeze(-1)
        cos_output = context_output*cos
        outputs = torch.cat([context_output, cos_output], dim=-1)

        return outputs
    
    def forward(self, sentence, img_obj):
        batch_size = sentence.shape[0]

        all_encoder_layers, attention_probs = self.mybert(img_obj, sentence)
        cls_token = torch.cat([
            # all_encoder_layers[-3][:, :1, :], 
            all_encoder_layers[-2][:, :1, :], all_encoder_layers[-1][:, :1, :], 
        ], dim=-1)

        cls_scores = self.attr_cls(cls_token)
        cls_scores = cls_scores.view(cls_scores.shape[0], cls_scores.shape[-1])
        cls_scores_active = self.sigmoid(cls_scores)
        
        batch_idxs, word_idxs = torch.where(cls_scores_active > 0.5)
        # no class predicted from cls stage
        if len(batch_idxs) == 0: return cls_scores, batch_idxs, word_idxs, torch.Tensor([])

        context_output = all_encoder_layers[-1]
        context_output = context_output[batch_idxs]

        selected_words = self.vocab_labels[word_idxs].long().cuda()
        
        attr_all_encoder_layers, _ = self.bert(selected_words)
        attr_token = attr_all_encoder_layers[:, :1, :]

        outputs = self.calculate_cosin(context_output, attr_token)
        outputs, _ = self.lstm(outputs)

        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)

        return cls_scores, batch_idxs, word_idxs, outputs
