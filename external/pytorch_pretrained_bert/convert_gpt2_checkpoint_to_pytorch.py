"""Convert OpenAI GPT checkpoint."""

from __future__ import absolute_import, division, print_function

import argparse
from io import open

import torch

from external.pytorch_pretrained_bert.modeling_gpt2 import (CONFIG_NAME, WEIGHTS_NAME,
                                                            GPT2Config,
                                                            GPT2Model,
                                                            load_tf_weights_in_gpt2)


def convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path):
    # Construct uls_dram
    if gpt2_config_file == "":
        config = GPT2Config()
    else:
        config = GPT2Config(gpt2_config_file)
    model = GPT2Model(config)

    # Load weights from numpy
    load_tf_weights_in_gpt2(model, gpt2_checkpoint_path)

    # Save pytorch-uls_dram
    pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
    print("Save PyTorch uls_dram to {}".format(pytorch_weights_dump_path))
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print("Save configuration file to {}".format(pytorch_config_dump_path))
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--gpt2_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the TensorFlow checkpoint path.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch uls_dram.")
    parser.add_argument("--gpt2_config_file",
                        default = "",
                        type = str,
                        help = "An optional config json file corresponding to the pre-trained OpenAI uls_dram. \n"
                            "This specifies the uls_dram architecture.")
    args = parser.parse_args()
    convert_gpt2_checkpoint_to_pytorch(args.gpt2_checkpoint_path,
                                         args.gpt2_config_file,
                                         args.pytorch_dump_folder_path)
