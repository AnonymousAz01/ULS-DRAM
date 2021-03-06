import torch
import torch.nn as nn
from external.pytorch_pretrained_bert import BertTokenizer
from uls_dram.module import Module
from uls_dram.visual_linguistic_bert import VisualLinguisticBert
from torchvision.models import resnet152

from paths import PathConfig
path_cfg = PathConfig()

class PretrainedEncoderUlsVL(Module):
    def __init__(self, config):

        super(PretrainedEncoderUlsVL, self).__init__(config)
        self.config = config
        print('load resnet152 pretrained uls-vl encoder')

        self.pre_resnet = resnet152()
        # NOTE: Pretrained ResNet only jointly finetuned during Training
        # self.pre_resnet.load_state_dict(torch.load(path_cfg.RESNET_152)) 
        
        self.pre_resnet = nn.Sequential(*list(self.pre_resnet.children())[:-2])

        self.object_visual_embeddings = nn.Linear(2048, config.NETWORK.VLBERT.hidden_size)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(path_cfg.ROBERTA_SMALL_CHINESE)
        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT)

        # init weights
        self.init_weight()

    def init_weight(self):
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)

    def fix_params(self):
        for param in self.pre_resnet.parameters():
            param.requires_grad = False
        for param in self.vlbert.parameters():
            param.requires_grad = False

    def forward(self,
                image,
                expression,
                relation=None,
                output_visual=False):

        batch_size = expression.shape[0]
        images = image

        img_feature = self.pre_resnet(images)
        img_feature = img_feature.view(batch_size, 2048, 7 * 7).transpose(2, 1)

        box_mask = torch.ones((batch_size, 49), dtype=torch.bool, device=expression.device)

        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1]))
        text_input_ids[:, :] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = torch.zeros(
            (text_input_ids.shape[0], text_input_ids.shape[1], self.config.NETWORK.VLBERT.hidden_size),
            device=expression.device,
        )
        object_visual_embedding = self.object_visual_embeddings(img_feature)
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            box_mask.new_zeros((box_mask.shape[0], box_mask.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((object_visual_embedding, object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_regions, _, attention_probs = self.vlbert(text_input_ids,
                                                                   text_token_type_ids,
                                                                   text_visual_embeddings,
                                                                   text_mask,
                                                                   object_vl_embeddings,
                                                                   box_mask,
                                                                #    output_all_encoded_layers=False,
                                                                   output_all_encoded_layers=True,
                                                                   output_text_and_object_separately=True,
                                                                   relation=relation)
        if output_visual:
            return hidden_states_text, hidden_states_regions, attention_probs
        else:
            return hidden_states_text, attention_probs
