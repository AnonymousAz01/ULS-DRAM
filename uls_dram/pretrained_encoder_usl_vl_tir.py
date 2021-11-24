from pytorch_pretrained_bert.modeling import BertEmbeddings, BertEncoder, BertPooler, BertPreTrainedModel
import torch
import torch.nn as nn
from uls_dram.module import Module
from torchvision.models import resnet152
from paths import PathConfig

path_cfg = PathConfig()

class TextGuidedInfoRangeMinimization(BertPreTrainedModel):
    
    def __init__(self, config):
        super(TextGuidedInfoRangeMinimization, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.internal_num_attention_heads = 1
        self.txt_hidden_size = config.hidden_size
        self.img_hidden_size = 2048
        self.attn_size = 200
        self.device = 'cuda'

        self.hiddenmm2embs = nn.Linear(self.attn_size, self.txt_hidden_size)

        self.W_q_txt = nn.Parameter(torch.randn(self.txt_hidden_size, self.attn_size, device=self.device) / 100, requires_grad=config.gradient_checkpointing)  # D1 * D
        self.W_k_txt = nn.Parameter(torch.randn(self.txt_hidden_size, self.attn_size, device=self.device) / 100, requires_grad=config.gradient_checkpointing) # D1 * D
        self.W_v_txt = nn.Parameter(torch.randn(self.txt_hidden_size, self.attn_size, device=self.device) / 100, requires_grad=config.gradient_checkpointing) # D1 * D

        self.W_q_img = nn.Parameter(torch.randn(self.txt_hidden_size, self.attn_size, device=self.device) / 100, requires_grad=config.gradient_checkpointing) # D1 * D
        self.W_k_img = nn.Parameter(torch.randn(self.img_hidden_size, self.attn_size, device=self.device) / 100, requires_grad=config.gradient_checkpointing) # D2 * D
        self.W_v_img = nn.Parameter(torch.randn(self.img_hidden_size, self.attn_size, device=self.device) / 100, requires_grad=config.gradient_checkpointing) # D2 * D

        self.apply(self.init_bert_weights)

    def cross_attention(self, txt_embedding, img_embedding):
        S1 = txt_embedding.shape[1]
        S2 = img_embedding.shape[1]

        # txt_attn
        q_matrix = torch.reshape(torch.matmul(torch.reshape(txt_embedding, (-1, self.txt_hidden_size)), self.W_q_txt),
                                 (-1, S1, self.attn_size))  # B * S1 * D
        k_matrix = torch.reshape(torch.matmul(torch.reshape(txt_embedding, (-1, self.txt_hidden_size)), self.W_k_txt),
                                 (-1, S1, self.attn_size))  # B * S1 * D
        v_matrix = torch.reshape(torch.matmul(torch.reshape(txt_embedding, (-1, self.txt_hidden_size)), self.W_v_txt),
                                 (-1, S1, self.attn_size))  # B * S1 * D

        q_matrix = torch.unsqueeze(q_matrix, 2)  # B * S1 * 1 * D
        k_matrix = torch.unsqueeze(k_matrix, 1)  # B * 1 * S1 * D

        qk = torch.div(torch.sum(q_matrix * k_matrix, dim=-1), torch.sqrt(torch.tensor(self.attn_size, dtype=torch.float32)))  # B * S1 * S1
        hiddens_txt = torch.matmul(qk, v_matrix) # B * S1 * D
        
        # img_attn
        q_matrix = torch.reshape(torch.matmul(torch.reshape(txt_embedding, (-1, self.txt_hidden_size)), self.W_q_img),
                                    (-1, S1, self.attn_size))  # B * S1 * D
        k_matrix = torch.reshape(torch.matmul(torch.reshape(img_embedding, (-1, self.img_hidden_size)), self.W_k_img),
                                    (-1, S2, self.attn_size))  # B * S2 * D
        v_matrix = torch.reshape(torch.matmul(torch.reshape(img_embedding, [-1, self.img_hidden_size]), self.W_v_img),
                                    (-1, S2, self.attn_size))  # B * S2 * D
        q_matrix = torch.unsqueeze(q_matrix, 2)  # B * S1 * 1 * D
        k_matrix = torch.unsqueeze(k_matrix, 1)  # B * 1 * S2 * D
        qk = torch.div(torch.sum(q_matrix * k_matrix, dim=-1), torch.sqrt(torch.tensor(self.attn_size,
                                                                                            dtype=torch.float32)))
        img_qk = qk
        hiddens_img = torch.bmm(qk, v_matrix)

        return self.hiddenmm2embs(hiddens_txt + hiddens_img), img_qk

    def forward(self, input_ids, visual_embedding,
        token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        batch_size, seqlen = input_ids.shape[:2]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Encode visual feature into text feature
        linguistic_embedding = self.embeddings(input_ids, token_type_ids)
        linguistic_embedding = linguistic_embedding

        visual_embedding = visual_embedding.view(batch_size, -1, self.img_hidden_size)

        visual_lingustic_embedding, visual_qk = self.cross_attention(linguistic_embedding, visual_embedding)

        visual_lingustic_embedding = visual_lingustic_embedding + linguistic_embedding

        encoded_layers = self.encoder(visual_lingustic_embedding,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]

        return { 
            'last_hidden_state': sequence_output, 
            'all_hidden_states': encoded_layers,
            'vl_embedding': visual_lingustic_embedding,
            'visual_qk': visual_qk,
        }

# external class for Text-Guided Information Range Minimization, 
# which are integrated inside customized pretrained bertmodel class TextGuidedInfoRangeMinimization
# and a jointly fine-tuned pretrained ResNet encoder
class PretrainedEncoderTIR(Module):
    def __init__(self, config, bert_model_name=path_cfg.BERT_BASE_UNCASED):

        super(PretrainedEncoderTIR, self).__init__(config)
        self.config = config
        print('load resnet152 pretrained encoder')
        self.hidden_size = self.config.NETWORK.VLBERT.hidden_size
        self.internal_num_attention_heads = 1
        self.internal_hidden_size = 200
        self.img_hidden_size = 2048

        self.pre_resnet = resnet152()
        # NOTE: Pretrained ResNet only jointly finetuned during Training
        # self.pre_resnet.load_state_dict(torch.load(path_cfg.RESNET_152)) 
        self.pre_resnet = nn.Sequential(*list(self.pre_resnet.children())[:-2])

        self.bert:TextGuidedInfoRangeMinimization = TextGuidedInfoRangeMinimization.from_pretrained(bert_model_name)

    def fix_params(self):
        for param in self.pre_resnet.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, image, expression):
        img_feature = self.pre_resnet(image)
        output = self.bert(expression, img_feature)
        return output['all_hidden_states'], output['visual_qk']