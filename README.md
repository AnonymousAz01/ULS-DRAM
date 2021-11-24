# ULS-DRAM
Improving Multi-Modal E-Commerce Attribute Value Extraction via Unified Learning Scheme and Dynamic Range Minimization

## STEP 1: Environment Setup

* Python 3.6.8
* CUDA 10.1
* cudnn 7.6.2
* PyTorch 1.6.0
* TorchVision 0.7.0
* NumPy 1.19.5
* pandas 1.1.5
* Pillow 8.4.0

1. Install the non-Python dependences including Nvidia GPU Driver, CUDA and Python. And 1x Nvidia V100 GPU (32G Version) is required for evaluation.
2. Run ``pip3 install -r requirements`` for install Python dependences.

## STEP 2: Prepare the Data and Pretrained Models
1. Download the datasets:
* MEPAVE: Following the instruction of official repository https://github.com/jd-aig/JAVE to fillup the application form, and then E-mail it to the contact author for the accessment of this dataset.

* MAE: Download the annotations of train and val subsets on official website https://rloganiv.github.io/mae/. Due to the zipped image files are corrupted, we have contacted the authors and obtained the new download link: https://drive.google.com/file/d/1g5a2NiKreNJx5fR1__8HraBoDkPQXR6Z/view?usp=sharing

2. Place the unarchived dataset files on path wherever you like, and modify the global paths in ``paths.py``. For the tokenization of input texts during inference time, pretrained BERT models are needed, please refer to the links in ``paths.py`` to download and modify the corresponding paths. (p.s. ResNet is only used for joinly fine-tuning during training)

3. For MAE dataset, an extra pre-processing is needed to convert the annotation from attribure-value pairs to "B-I-O" sequences. Run ``python3 convert_mae.py`` for this conversion.

## STEP 3: Evaluate the models
1. Run ``python3 test_mae.py`` to obtain the evaluation results of ULS-DRAM on MAE benckmark.

2. Run ``python3 test_mepave.py`` to obtain the evaluation results of ULS-DRAM on MEPAVE benckmark.

You will get the final output as follows: 

MAE:
```
CLS eval results: Precision, Recall, F1
                0.83447	0.80507	0.81951

TAG eval results: Precision, Recall, F1
                0.85429 0.78038 0.81567

Accuracy score: 79.20237
```

MEPAVE:
```
CLS eval results: Precision, Recall, F1
                0.97285 0.98024 0.97653

TAG eval results: Precision, Recall, F1
                0.95887 0.97862 0.96864
```
