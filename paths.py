# Paths are reset for anonymous

class PathConfig:
    # Dataset Path
    MEPAVE_PATH = '/path/to/mepave/corpus'
    MAE_PATH = '/path/to/mae/'

    # Pretrained Models
    BERT_BASE_UNCASED = '/path/to/bert-base-uncased/'
    # https://huggingface.co/bert-base-uncased

    ROBERTA_SMALL_CHINESE = '/path/to/chinese_roberta_L-4_H-512/'
    # https://huggingface.co/uer/chinese_roberta_L-4_H-512

    RESNET_152 = '/path/to/resnet152-b121ed2d.pth'
    # https://download.pytorch.org/models/resnet152-b121ed2d.pth
    
    # Checkpoints of Our Models
    MAE_MODEL_PATH = '/path/to/ULS-DRAM/checkpoints/mae/epoch38.pth'
    MEPAVE_MODEL_PATH = '/path/to/ULS-DRAM/checkpoints/mepave/epoch45.pth'
    