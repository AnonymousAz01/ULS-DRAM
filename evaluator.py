from typing import Dict
from seqeval.reporters import DictReporter
from data_loader_mae import MAEExtractorDataLoader
from data_loader_mepave import MEPAVEExtractorDataLoader
from uls_dram.extractors import UlsDramMAE, UlsDramMEPAVE
from util import *
from torchcrf import CRF
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report

class MAEExtractorEvaluator:
    def __init__(self, params, data_loader: MAEExtractorDataLoader):
        self.params = params
        self.ex_dl:MAEExtractorDataLoader = data_loader

    def cls_postprocess(self, cls_scores, y):
        pred = nn.Sigmoid()(cls_scores)

        preds_batch = pred.squeeze(1).detach().cpu().numpy().tolist()
        ground_batch = y.numpy().tolist()

        th = 0.5
        preds_attrs = []
        for pred_score_list in preds_batch:
            preds_label = [1 if pred_score > th else 0 for pred_score in pred_score_list]
            preds_attrs.append(preds_label)
        
        return ground_batch, preds_attrs

    def tag_postprocess_prec_recall_f1(self, ground_seqs, ground_scores, pred_batch_idxs, pred_word_idxs, pred_seqs, mask, loss_function_crf:CRF, cls_sep_token_len=0):
        batch_size = ground_scores.shape[0]
        sent_maxlen = self.ex_dl.sent_maxlen + cls_sep_token_len

        ground_indexs = [ [] for _ in range(batch_size) ]
        ground_indexs_map = np.nonzero(ground_scores.numpy())
        for idx, value in zip(*ground_indexs_map):
            ground_indexs[idx] += [value]

        new_ground_seqs = [ [] for _ in range(batch_size) ]
        new_pred_seqs = [ [] for _ in range(batch_size) ]

        if len(pred_batch_idxs) == 0:
            pred_seqs = []
            for batch_idx, cur_ground_indexs in enumerate(ground_indexs):
                for cur_ground_index in cur_ground_indexs:
                    new_ground_seqs[batch_idx].append(ground_seqs[batch_idx][ground_indexs[batch_idx].index(cur_ground_index)].tolist())
                    new_pred_seqs[batch_idx].append([self.ex_dl.tags2id['O'] if valid and ii != 0 else self.ex_dl.tags2id[''] for ii, valid in enumerate(mask[batch_idx].bool())])

        else:
            pred_seqs = pred_seqs.cuda()
            crf_mask = mask[pred_batch_idxs].bool().cuda()
            pred_seqs = loss_function_crf.decode(pred_seqs, crf_mask)
            pred_seqs = [ seq+[self.ex_dl.tags2id['']] * (sent_maxlen-len(seq)) for seq in pred_seqs ]

            pred_indexs = [ [] for _ in range(batch_size) ]
            for idx, value in zip(pred_batch_idxs, pred_word_idxs):
                pred_indexs[idx] += [value]

            for idx, value in zip(pred_batch_idxs, pred_seqs):
                new_pred_seqs[idx] += [value]

            for batch_idx, (cur_pred_indexs, cur_ground_indexs) in enumerate(zip(pred_indexs, ground_indexs)):
                for cur_pred_index in cur_pred_indexs:
                    # pred true & ground true: True Positive
                    if cur_pred_index in cur_ground_indexs: 
                        new_ground_seqs[batch_idx].append(ground_seqs[batch_idx][ground_indexs[batch_idx].index(cur_pred_index)].tolist())
                    # pred true & ground false: False Positive
                    else:
                        new_ground_seqs[batch_idx].append([ self.ex_dl.tags2id['O'] if valid and ii != 0 else self.ex_dl.tags2id[''] for ii, valid in enumerate(mask[batch_idx].bool()) ])
                    
                # TODO: pred false & ground true: False Negative
                for cur_ground_index in cur_ground_indexs:
                    if cur_ground_index not in cur_pred_indexs:
                        new_ground_seqs[batch_idx].append(ground_seqs[batch_idx][ground_indexs[batch_idx].index(cur_ground_index)].tolist())
                        new_pred_seqs[batch_idx].append([ self.ex_dl.tags2id['O'] if valid and ii != 0 else self.ex_dl.tags2id[''] for ii, valid in enumerate(mask[batch_idx].bool()) ])
        
        tmp_new_ground_seqs = []
        for cur_new_ground_seqs in new_ground_seqs:
            if len(cur_new_ground_seqs) > 0: tmp_new_ground_seqs.extend(cur_new_ground_seqs)
        new_ground_seqs = tmp_new_ground_seqs

        tmp_new_pred_seqs = []
        for cur_new_pred_seqs in new_pred_seqs:
            if len(cur_new_pred_seqs) > 0: tmp_new_pred_seqs.extend(cur_new_pred_seqs)
        new_pred_seqs = tmp_new_pred_seqs
        
        new_ground_seqs = [ [ self.ex_dl.id2tags[int(item)].replace('X', 'I') if int(item) != self.ex_dl.tags2id[''] else 'O' for item in seq ] for seq in new_ground_seqs ]
        new_pred_seqs = [ [ self.ex_dl.id2tags[int(item)].replace('X', 'I') if int(item) != self.ex_dl.tags2id[''] else 'O' for item in seq ] for seq in new_pred_seqs ]

        assert all([ len(seq) == sent_maxlen for seq in new_ground_seqs]) and all([ len(seq) == sent_maxlen for seq in new_pred_seqs])
        return new_ground_seqs, new_pred_seqs

    def tag_postprocess_accuracy(self, ground_seqs, ground_scores, pred_batch_idxs, pred_word_idxs, pred_seqs, mask, loss_function_crf:CRF, ground_sents, cls_sep_token_len=0):
        batch_size = ground_scores.shape[0]
        sent_maxlen = self.ex_dl.sent_maxlen + cls_sep_token_len

        ground_indexs = [ [] for _ in range(batch_size) ]
        ground_sents = [ self.ex_dl.tokenizer.convert_ids_to_tokens(sent) for sent in ground_sents ]

        ground_indexs_map = np.nonzero(ground_scores.numpy())
        for idx, value in zip(*ground_indexs_map):
            ground_indexs[idx] += [value]

        new_ground_seqs = [ [] for _ in range(batch_size) ]
        new_pred_seqs = [ [] for _ in range(batch_size) ]
        new_ground_sents = [ [] for _ in range(batch_size) ]

        if len(pred_batch_idxs) == 0:
            pred_seqs = []
            for batch_idx, cur_ground_indexs in enumerate(ground_indexs):
                for cur_ground_index in cur_ground_indexs:
                    new_ground_seqs[batch_idx].append(ground_seqs[batch_idx][ground_indexs[batch_idx].index(cur_ground_index)].tolist())
                    new_pred_seqs[batch_idx].append([self.ex_dl.tags2id['O'] if valid and ii != 0 else self.ex_dl.tags2id[''] for ii, valid in enumerate(mask[batch_idx].bool())])
                    new_ground_sents[batch_idx].append(ground_sents[batch_idx])
        else:
            pred_seqs = pred_seqs.cuda()
            crf_mask = mask[pred_batch_idxs].bool().cuda()
            pred_seqs = loss_function_crf.decode(pred_seqs, crf_mask)
            pred_seqs = [ seq+[self.ex_dl.tags2id['']] * (sent_maxlen-len(seq)) for seq in pred_seqs ]

            # pred seqs with key of batch_idx
            tmp_cur_pred_seqs = [ [] for _ in range(batch_size) ]
            for idx, value in zip(pred_batch_idxs, pred_seqs):
                tmp_cur_pred_seqs[idx] += [value]
            pred_seqs = tmp_cur_pred_seqs
            
            # pred attr_idxs with key of batch_idx
            tmp_pred_indexs = [ [] for _ in range(batch_size) ]
            for idx, value in zip(pred_batch_idxs, pred_word_idxs):
                tmp_pred_indexs[idx] += [value]
            pred_indexs = tmp_pred_indexs

            for batch_idx, (cur_ground_indexs, cur_pred_indexs) in enumerate(zip(ground_indexs, pred_indexs)):
                for cur_ground_index in cur_ground_indexs:
                    if cur_ground_index in cur_pred_indexs:
                        # TP
                        new_pred_seqs[batch_idx].append(pred_seqs[batch_idx][pred_indexs[batch_idx].index(cur_ground_index)])
                    else:
                        # FN, add empty prediction
                        new_pred_seqs[batch_idx].append([ self.ex_dl.tags2id['O'] if valid and ii != 0 else self.ex_dl.tags2id[''] for ii, valid in enumerate(mask[batch_idx].bool()) ])

                    new_ground_seqs[batch_idx].append(ground_seqs[batch_idx][ground_indexs[batch_idx].index(cur_ground_index)].tolist())
                    new_ground_sents[batch_idx].append(ground_sents[batch_idx])

        # Flatten them all to [-1, max_sent_len]
        tmp_new_ground_seqs = []
        for cur_new_ground_seqs in new_ground_seqs:
            if len(cur_new_ground_seqs) > 0: tmp_new_ground_seqs.extend(cur_new_ground_seqs)
        new_ground_seqs = tmp_new_ground_seqs

        tmp_new_pred_seqs = []
        for cur_new_pred_seqs in new_pred_seqs:
            if len(cur_new_pred_seqs) > 0: tmp_new_pred_seqs.extend(cur_new_pred_seqs)
        new_pred_seqs = tmp_new_pred_seqs

        tmp_new_ground_sents = []
        for cur_new_ground_sents in new_ground_sents:
            if len(cur_new_ground_sents) > 0: tmp_new_ground_sents.extend(cur_new_ground_sents)
        new_ground_sents = tmp_new_ground_sents
        
        # replace X to I, and '' to O
        new_ground_seqs = [ [ self.ex_dl.id2tags[int(item)].replace('X', 'I') if int(item) != self.ex_dl.tags2id[''] else 'O' for item in seq ] for seq in new_ground_seqs ]
        new_pred_seqs = [ [ self.ex_dl.id2tags[int(item)].replace('X', 'I') if int(item) != self.ex_dl.tags2id[''] else 'O' for item in seq ] for seq in new_pred_seqs ]

        # Assertion
        assert len(new_ground_seqs) == len(new_ground_sents) == len(new_ground_seqs) == len(ground_indexs_map[0])
        assert all([ len(seq) == sent_maxlen for seq in new_ground_seqs]) and \
            all([ len(seq) == sent_maxlen for seq in new_pred_seqs]) and \
            all([ len(sent) == sent_maxlen for sent in new_ground_sents])
        return new_ground_seqs, new_pred_seqs, new_ground_sents
    
    def extract_kvpairs_in_bio(self, bio_seq, word_seq):
        assert len(bio_seq) == len(word_seq)
        pairs = set()
        pre_bio = "O"
        v = ""
        for i, bio in enumerate(bio_seq):
            if (bio == "O"):
                if v != "": pairs.add((pre_bio[2:], v))
                v = ""
            elif (bio[0] == "B"):
                if v != "": pairs.add((pre_bio[2:], v))
                v = word_seq[i]
            elif (bio[0] == "I"):
                if (pre_bio[0] == "O") or (pre_bio[2:] != bio[2:]):
                    if v != "": pairs.add((pre_bio[2:], v))
                    v = ""
                else:
                    v += word_seq[i]
            pre_bio = bio
        if v != "": pairs.add((pre_bio[2:], v))
        return pairs

    def compute_accuracy_pairs(self, ground_seqs, pred_seqs, ground_sents):
        correct_value_count, total_value_count = 0, 0
        for ground_seq, pred_seq, ground_sent in zip(ground_seqs, pred_seqs, ground_sents):
            ground_values = self.extract_kvpairs_in_bio(ground_seq, ground_sent)
            ground_values = [ item[1] for item in list(ground_values) ]

            pred_values = self.extract_kvpairs_in_bio(pred_seq, ground_sent)
            pred_values = [ item[1] for item in list(pred_values) ]

            total_value_count += len(ground_values)
            if sorted(ground_values) == sorted(pred_values): 
                correct_value_count += len(ground_values)
        
        return correct_value_count, total_value_count

    def get_accuracy_mae(self, model:UlsDramMAE, split):
        if split == 'valid':
            data_loader = self.ex_dl.valid_dataloader
        else:
            raise ModuleNotFoundError('No such subset:', split)

        model.eval()
        ground_attrs_total, preds_attrs_total = [], []
        ground_seqs_total, preds_seqs_total = [], []
        ground_seqs_acc_total, preds_seqs_acc_total = [], []
        correct_value_count_total, total_value_count_total = 0, 0

        print('Currently under evaluation on %s set: ' % split)
        for ii, (x, obj_x, y, tag_seqs, mask, mcls_tag_seq, ocls_tag_seq) in enumerate(data_loader):
            torch.cuda.empty_cache()
            cls_scores, batch_idxs, word_idxs, seq_outputs = model(to_variable(x), to_variable(obj_x)) # seq_len * bs * labels
            ground_batch, preds_attrs = self.cls_postprocess(cls_scores, y)

            loss_function_crf:CRF= model.module.loss_function_crf if 'local_rank' in self.params else model.loss_function_crf
            
            ground_sents = x.tolist()
            ground_seqs, pred_seqs, ground_sents = self.tag_postprocess_accuracy(tag_seqs, y, batch_idxs, word_idxs, seq_outputs, mask, loss_function_crf, ground_sents)
            correct_value_count, total_value_count = self.compute_accuracy_pairs(ground_seqs, pred_seqs, ground_sents)

            ground_seqs_acc_total += ground_seqs
            preds_seqs_acc_total += pred_seqs
            correct_value_count_total += correct_value_count
            total_value_count_total += total_value_count

            ground_seqs, pred_seqs = self.tag_postprocess_prec_recall_f1(tag_seqs, y, batch_idxs, word_idxs, seq_outputs, mask, loss_function_crf)
            ground_seqs_total += ground_seqs
            preds_seqs_total += pred_seqs

            ground_attrs_total += ground_batch
            preds_attrs_total += preds_attrs

            if ii % self.params.print_freq_eval == 0:
                print('Iteration processed: %04d / %04d' % (ii, len(data_loader)))

        print('Processing done, start evaluation...')

        micro_avg = precision_recall_fscore_support(ground_attrs_total, preds_attrs_total, average='micro')
        macro_avg = precision_recall_fscore_support(ground_attrs_total, preds_attrs_total, average='macro', zero_division=0)

        # NOTE: accuracy_score acc = #(absolutely the same sequence for each attribute)/#(attribute count)
        report : DictReporter = classification_report(ground_seqs_total, preds_seqs_total, output_dict=True, digits=4)
        micro_avg_report:Dict = report['micro avg']
        precision, recall, f1 = micro_avg_report['precision'], micro_avg_report['recall'], micro_avg_report['f1-score']

        accuracy = correct_value_count_total * 100 / total_value_count_total
        
        print('\nCLS eval results: Precision, Recall, F1')
        print('\t\t'+'\t'.join(map(lambda x: str(round(float(x), 5)), micro_avg[:-1])))

        print('\nTAG eval results: Precision, Recall, F1')
        print('\t\t'+'\t'.join([str(round(float(x), 5)) for x in [precision, recall, f1]]))    
        
        print('Accuracy score:\t'+str(round(float(accuracy), 5)))

        return [*micro_avg[:-1], precision, recall, f1, accuracy]

class MEPAVEExtractorEvaluator:
    def __init__(self, params, data_loader: MEPAVEExtractorDataLoader):
        self.params = params
        self.ex_dl: MEPAVEExtractorDataLoader = data_loader

    def cls_postprocess(self, cls_scores, y):
        pred = nn.Sigmoid()(cls_scores)

        preds_batch = pred.squeeze(1).detach().cpu().numpy().tolist()
        ground_batch = y.numpy().tolist()

        th = 0.5
        preds_attrs = []
        for pred_score_list in preds_batch:
            preds_label = [1 if pred_score > th else 0 for pred_score in pred_score_list]
            preds_attrs.append(preds_label)
        
        return ground_batch, preds_attrs
 
    def tag_postprocess_unfix_len(self, ground_seqs, ground_scores, pred_batch_idxs, pred_word_idxs, pred_seqs, mask, loss_function_crf:CRF, cls_sep_token_len=0):
        batch_size = ground_scores.shape[0]
        sent_maxlen = self.ex_dl.sent_maxlen + cls_sep_token_len

        ground_indexs = [ [] for _ in range(batch_size) ]
        ground_indexs_map = np.nonzero(ground_scores.numpy())
        for idx, value in zip(*ground_indexs_map):
            ground_indexs[idx] += [value]

        new_ground_seqs = [ [] for _ in range(batch_size) ]
        new_pred_seqs = [ [] for _ in range(batch_size) ]

        if len(pred_batch_idxs) == 0:
            pred_seqs = []
            for batch_idx, cur_ground_indexs in enumerate(ground_indexs):
                for cur_ground_index in cur_ground_indexs:
                    new_ground_seqs[batch_idx].append(ground_seqs[batch_idx][ground_indexs[batch_idx].index(cur_ground_index)].tolist())
                    new_pred_seqs[batch_idx].append([self.ex_dl.tags2id['O'] if valid and ii != 0 else self.ex_dl.tags2id[''] for ii, valid in enumerate(mask[batch_idx].bool())])

        else:
            pred_seqs = pred_seqs.cuda()
            crf_mask = mask[pred_batch_idxs].bool().cuda()
            pred_seqs = loss_function_crf.decode(pred_seqs, crf_mask)
            pred_seqs = [ seq+[self.ex_dl.tags2id['']] * (sent_maxlen-len(seq)) for seq in pred_seqs ]

            pred_indexs = [ [] for _ in range(batch_size) ]
            for idx, value in zip(pred_batch_idxs, pred_word_idxs):
                pred_indexs[idx] += [value]

            for idx, value in zip(pred_batch_idxs, pred_seqs):
                new_pred_seqs[idx] += [value]

            for batch_idx, (cur_pred_indexs, cur_ground_indexs) in enumerate(zip(pred_indexs, ground_indexs)):
                for cur_pred_index in cur_pred_indexs:
                    # pred true & ground true: True Positive
                    if cur_pred_index in cur_ground_indexs: 
                        new_ground_seqs[batch_idx].append(ground_seqs[batch_idx][ground_indexs[batch_idx].index(cur_pred_index)].tolist())
                    # pred true & ground false: False Positive
                    else:
                        new_ground_seqs[batch_idx].append([ self.ex_dl.tags2id['O'] if valid and ii != 0 else self.ex_dl.tags2id[''] for ii, valid in enumerate(mask[batch_idx].bool()) ])
                    
                # TODO: pred false & ground true: False Negative
                for cur_ground_index in cur_ground_indexs:
                    if cur_ground_index not in cur_pred_indexs:
                        new_ground_seqs[batch_idx].append(ground_seqs[batch_idx][ground_indexs[batch_idx].index(cur_ground_index)].tolist())
                        new_pred_seqs[batch_idx].append([ self.ex_dl.tags2id['O'] if valid and ii != 0 else self.ex_dl.tags2id[''] for ii, valid in enumerate(mask[batch_idx].bool()) ])
        
        tmp_new_ground_seqs = []
        for cur_new_ground_seqs in new_ground_seqs:
            if len(cur_new_ground_seqs) > 0: tmp_new_ground_seqs.extend(cur_new_ground_seqs)
        new_ground_seqs = tmp_new_ground_seqs

        tmp_new_pred_seqs = []
        for cur_new_pred_seqs in new_pred_seqs:
            if len(cur_new_pred_seqs) > 0: tmp_new_pred_seqs.extend(cur_new_pred_seqs)
        new_pred_seqs = tmp_new_pred_seqs
        
        new_ground_seqs = [ [ self.ex_dl.id2tags[int(item)].replace('X', 'I') if int(item) != self.ex_dl.tags2id[''] else 'O' for item in seq ] for seq in new_ground_seqs ]
        new_pred_seqs = [ [ self.ex_dl.id2tags[int(item)].replace('X', 'I') if int(item) != self.ex_dl.tags2id[''] else 'O' for item in seq ] for seq in new_pred_seqs ]

        assert all([ len(seq) == sent_maxlen for seq in new_ground_seqs]) and all([ len(seq) == sent_maxlen for seq in new_pred_seqs])
        return new_ground_seqs, new_pred_seqs

    def get_accuracy(self, model: UlsDramMEPAVE, split):
        if split == 'valid':
            data_loader = self.ex_dl.valid_dataloader
        else:
            data_loader = self.ex_dl.test_dataloader

        model.eval()
        ground_attrs_total, preds_attrs_total = [], []
        ground_seqs_total, preds_seqs_total = [], []

        print('Currently under evaluation on %s set: ' % split)
        for x, obj_x, y, tag_seqs, mask, mcls_tag_seq, ocls_tag_seq in data_loader:

            cls_scores, batch_idxs, word_idxs, seq_outputs = model(to_variable(x), to_variable(obj_x)) # seq_len * bs * labels
            # cls_scores, batch_idxs, word_idxs, seq_outputs, ocls_seq_output = model(to_variable(x), to_variable(obj_x)) # seq_len * bs * labels
            ground_batch, preds_attrs = self.cls_postprocess(cls_scores, y)

            ground_seqs, pred_seqs = self.tag_postprocess_unfix_len(tag_seqs, y, batch_idxs, word_idxs, seq_outputs, mask, model.loss_function_crf, cls_sep_token_len=2)

            ground_attrs_total += ground_batch
            preds_attrs_total += preds_attrs
            ground_seqs_total += ground_seqs
            preds_seqs_total += pred_seqs

        print('Processing done, start evaluation...')
        micro_avg = precision_recall_fscore_support(ground_attrs_total, preds_attrs_total, average='micro')
        macro_avg = precision_recall_fscore_support(ground_attrs_total, preds_attrs_total, average='macro', zero_division=0)

        # report = classification_report(ground_seqs_total, preds_seqs_total, digits=4)
        # print(report)

        report : DictReporter = classification_report(ground_seqs_total, preds_seqs_total, output_dict=True, digits=4)
        micro_avg_report:Dict = report['micro avg']
        precision, recall, f1 = micro_avg_report['precision'], micro_avg_report['recall'], micro_avg_report['f1-score']

        print('\nCLS eval results: Precision, Recall, F1')
        print('\t\t'+'\t'.join(map(lambda x: str(round(float(x), 5)), micro_avg[:-1])))

        print('\nTAG eval results: Precision, Recall, F1')
        print('\t\t'+'\t'.join([str(round(float(x), 5)) for x in [precision, recall, f1]]))    
        
        return [*micro_avg[:-1], *list(micro_avg_report.values())[:-1]]
