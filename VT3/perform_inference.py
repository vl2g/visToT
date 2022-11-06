import nltk
from metrics import eval_generated_text
nltk.download('stopwords')
nltk.download('punkt')
import os
import sys
import random
from math import ceil
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import operator
from operator import itemgetter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import yaml
import subprocess
from subprocess import call
from data_utils import reward_estimation, dev_corpus_bleu_estimation
import argparse
from dataclass import Data
import logging
from pprint import pprint
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_table_text_path', type=str)
    parser.add_argument('--train_image_data_path', type=str)
    parser.add_argument('--train_reference_sentence_path', type=str)
    parser.add_argument('--dev_table_text_path', type=str)
    parser.add_argument('--dev_image_data_path', type=str)
    parser.add_argument('--dev_reference_sentence_path', type=str)
    parser.add_argument('--dev_reference_path', type=str)
    parser.add_argument('--image_feat_path', type=str)
    parser.add_argument('--special_token_path', type=str)
    parser.add_argument('--max_table_len', type=int, default=320)
    parser.add_argument('--max_tgt_len', type=int, default=80)
    parser.add_argument('--min_slot_key_cnt', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--max_decode_len', type=int, default=80)
    parser.add_argument('--feat_dim', type=int, default=768)
    parser.add_argument('--pos_dim', type=int, default=4)
    parser.add_argument('--use_vis_layer_norm', type=str2bool, default=True)
    parser.add_argument('--individual_vis_layer_norm', type=str2bool, default=True)
    parser.add_argument('--share_vis_lang_layer_norm', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pretrained_ckpt_path', type=str)
    parser.add_argument('--output_path', type=str, default=r'./output/test_result/')
    parser.add_argument('--pretrain_mode', type=str2bool, default=False)
    return parser.parse_args()

def map_cuda(tensor_item, device, is_cuda):
    res_list = []
    if is_cuda:
        if isinstance(tensor_item, tuple) and len(tensor_item) == 2:
            res_list.append(tensor_item[0].cuda(device))
            res_list.append(tensor_item[1].cuda(device))
        elif isinstance(tensor_item, tuple) and len(tensor_item) == 1:
            res_list.append(tensor_item[0].cuda(device))
        else:
            assert 0
    else:
        res_list = tensor_item
    return res_list

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        file_log.write(data)
    def flush(self):
        pass

def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    original_keys = list(state_dict['model'].keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict['model'][new_key] = state_dict['model'].pop(key)
    return state_dict['model']


if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()

    args = parse_config()
    device = args.gpu_id

    test_output_dir = os.path.join(args.output_path, 'test_output')
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir, exist_ok=True)

    # setup logging
    run_log_counter = 0
    while(os.path.exists(test_output_dir + '/run_{}.log'.format(run_log_counter))):
        run_log_counter += 1
    file_log = open(test_output_dir + '/run_{}.log'.format(run_log_counter), 'w')
    file_log.write("")
    
    sys.stdout = Unbuffered(sys.stdout)
    print('Logging enabled', flush=True)

    print(args)

    print ('Start loading data...')
    train_dict, dev_dict = {}, {}
    train_dict['table_text_path'] = args.train_table_text_path
    train_dict['image_data_path'] = args.train_image_data_path
    train_dict['reference_sentence_path'] = args.train_reference_sentence_path

    dev_dict['table_text_path'] = args.dev_table_text_path
    dev_dict['image_data_path'] = args.dev_image_data_path
    dev_dict['reference_sentence_path'] = args.dev_reference_sentence_path
    special_token_name = args.special_token_path

    train_dict['processed_file_path'] = None
    dev_dict['processed_file_path'] = None

    data = Data(train_dict, dev_dict, args.image_feat_path, args.max_table_len, args.max_tgt_len, 
        args.model_name, args.special_token_path, args.min_slot_key_cnt, args.pretrain_mode)

    from mm_generator import MMGenerator
    model = MMGenerator(
        model_name=args.model_name,
        tokenizer=data.decode_tokenizer,
        max_decode_len=args.max_decode_len,
        dropout=0.0,
        feat_dim=args.feat_dim,
        use_vis_layer_norm=args.use_vis_layer_norm,
        individual_vis_layer_norm=args.individual_vis_layer_norm,
        share_vis_lang_layer_norm=args.share_vis_lang_layer_norm,
        )
    print('Model loaded')

    print ('Loading Pretrained Parameters...')
    if torch.cuda.is_available():
        state_dict = load_state_dict(args.pretrained_ckpt_path, 'cpu')
        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("vis_encoder."):
                new_key = 'encoder.' + key[len("vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith("model.vis_encoder."):
                new_key = 'model.encoder.' + key[len("model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

        results = model.load_state_dict(state_dict, strict=False)
        print('Model loaded from ', args.pretrained_ckpt_path)
        pprint(results)
    else:
        model_ckpt = torch.load(args.pretrained_ckpt_path, map_location='cpu')
        model_parameters = model_ckpt['model']
        model.load_state_dict(model_parameters)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda(device)
    model.eval()

    dev_num = data.dev_num
    batch_size = args.batch_size
    dev_step_num = ceil(dev_num / batch_size)

    dev_output_text_list = []
    print ('Start evaluation...')
    with torch.no_grad():
        for dev_step in tqdm(range(dev_step_num)):
            dev_batch_src_item, dev_batch_src_vis_inputs, dev_batch_tgt_item = data.get_next_dev_batch(dev_step, batch_size)

            dev_batch_src_vis_inputs = map_cuda(dev_batch_src_vis_inputs, device, cuda_available)
            dev_batch_src_tensor, dev_batch_src_mask = map_cuda(dev_batch_src_item, device, cuda_available)
            dev_batch_tgt_in_tensor, dev_batch_tgt_out_tensor = map_cuda(dev_batch_tgt_item, device, cuda_available)
            decoded_result = model.module.generate(dev_batch_src_tensor, dev_batch_src_mask, dev_batch_src_vis_inputs)
            dev_output_text_list += decoded_result

        dev_output_text_list = dev_output_text_list[:dev_num]
        dev_text_out_path = os.path.join(test_output_dir, 'inference_test_out.txt')
        with open(dev_text_out_path, 'w', encoding = 'utf8') as o:
            for text in dev_output_text_list:
                o.writelines(text + '\n')

        bleu_score = eval_generated_text(dev_text_out_path, args.dev_reference_sentence_path, args.dev_image_data_path)
