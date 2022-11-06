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
import argparse
from dataclass import Data
import time
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
    parser.add_argument('--image_feat_path', type=str)
    parser.add_argument('--special_token_path', type=str)
    parser.add_argument('--dev_reference_path', type=str)
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
    parser.add_argument('--total_steps', type=int, default=300000)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--print_every', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--output_path', type=str, default=r'./output/pretrain/')
    parser.add_argument('--pretrained_ckpt_path', type=str, default=None)
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

    test_output_dir = args.output_path
    val_results_dir = os.path.join(test_output_dir, 'val_results')
    if os.path.exists(test_output_dir):
        pass
    else: # recursively construct directory
        os.makedirs(test_output_dir, exist_ok=True)
    if os.path.exists(val_results_dir):
        pass
    else:
        os.makedirs(val_results_dir, exist_ok=True)

    # setup logging
    run_log_counter = 0
    while(os.path.exists(val_results_dir + '/run_{}.log'.format(run_log_counter))):
        run_log_counter += 1
    file_log = open(val_results_dir + '/run_{}.log'.format(run_log_counter), 'w')
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
        args.model_name, special_token_name, args.min_slot_key_cnt, args.pretrain_mode)
    print ('Data loaded.')

    from mm_generator import MMGenerator
    model = MMGenerator(
        model_name=args.model_name,
        tokenizer=data.decode_tokenizer,
        max_decode_len=args.max_decode_len,
        dropout=args.dropout,
        feat_dim=args.feat_dim,
        use_vis_layer_norm=args.use_vis_layer_norm,
        individual_vis_layer_norm=args.individual_vis_layer_norm,
        share_vis_lang_layer_norm=args.share_vis_lang_layer_norm,
        )

    print('Model loaded')

    if args.pretrained_ckpt_path: # Only load pretrained weights if we provide its path arg
        print ('Loading Pretrained Parameters...')
        if torch.cuda.is_available():
            # model_ckpt = torch.load(args.pretrained_ckpt_path)
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
    model.train()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01, # default weight decay value for AdamW
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_update_steps = (args.total_steps // args.gradient_accumulation_steps) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps)
    optimizer.zero_grad()

    train_num, dev_num = data.train_num, data.dev_num
    batch_size = args.batch_size
    train_step_num = int(train_num / batch_size) + 1 # number of steps (batches) to make an epoch in train/dev
    dev_step_num = ceil(dev_num / batch_size)

    batches_processed = 0
    max_dev_score = 0.
    total_steps = args.total_steps
    print_every, eval_every = args.print_every, args.eval_every

    train_loss_accumulated = 0. 

    start = time.time()
    model.train()
    for one_step in range(total_steps):
        epoch = one_step // train_step_num
        batches_processed += 1
        
        train_batch_src_item, train_batch_src_vis_inputs, train_batch_tgt_item = data.get_next_train_batch(batch_size)

        train_batch_src_vis_inputs = map_cuda(train_batch_src_vis_inputs, device, cuda_available)
        train_batch_src_tensor, train_batch_src_mask = map_cuda(train_batch_src_item, device, cuda_available)
        train_batch_tgt_in_tensor, train_batch_tgt_out_tensor = map_cuda(train_batch_tgt_item, device, cuda_available)

        train_loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_src_vis_inputs, train_batch_tgt_in_tensor, train_batch_tgt_out_tensor)
        train_loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_loss_accumulated += train_loss.sum().item()

        if (one_step+1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
                 
        if batches_processed % print_every == 0:
            curr_train_loss = train_loss_accumulated / print_every
            print ('At epoch %d, batch %d, train loss %.5f, max combine score is %5f' % 
                (epoch, batches_processed, curr_train_loss, max_dev_score))
            train_loss_accumulated = 0.   

        if batches_processed % eval_every == 0:
            model.eval()
            dev_loss_accumulated = 0.
            dev_output_text_list = []
            print ('Start evaluation...')
            with torch.no_grad():
                for dev_step in tqdm(range(dev_step_num)):
                    dev_batch_src_item, dev_batch_src_vis_inputs, dev_batch_tgt_item = data.get_next_dev_batch(dev_step, batch_size)
                    dev_batch_src_tensor, dev_batch_src_mask = map_cuda(dev_batch_src_item, device, cuda_available)
                    dev_batch_src_vis_inputs = map_cuda(dev_batch_src_vis_inputs, device, cuda_available)
                    dev_batch_tgt_in_tensor, dev_batch_tgt_out_tensor = map_cuda(dev_batch_tgt_item, device, cuda_available)
                    dev_loss = model(dev_batch_src_tensor, dev_batch_src_mask, dev_batch_src_vis_inputs, dev_batch_tgt_in_tensor, dev_batch_tgt_out_tensor)
                    dev_loss_accumulated += dev_loss.sum().item()

                    decoded_result = model.module.generate(dev_batch_src_tensor, dev_batch_src_mask, dev_batch_src_vis_inputs)
                    dev_output_text_list += decoded_result

                dev_output_text_list = dev_output_text_list[:dev_num]
                dev_text_out_path = os.path.join(val_results_dir, f'./test_out_{batches_processed}_epoch_{epoch}.txt')
                with open(dev_text_out_path, 'w', encoding = 'utf8') as o:
                    for text in dev_output_text_list:
                        o.writelines(text + '\n')

                one_dev_loss = dev_loss_accumulated / dev_step_num
                print ('----------------------------------------------------------------')
                print ('Validation: Epoch {}, batch {}, dev_loss is {:.5f}'.format(
                    epoch, batches_processed, one_dev_loss))
                bleu_score = eval_generated_text(dev_text_out_path, args.dev_reference_sentence_path, args.dev_image_data_path)

                one_dev_combine_score = bleu_score
                save_name = f'/checkpoint-{epoch}-{batches_processed}.ckpt'
                best_save_name = '/best-checkpoint.ckpt'

                if epoch % 2 == 0:
                    torch.save({'model':model.state_dict()}, test_output_dir + save_name)
                    print("Saving model weights: {}".format(save_name[1:]))

                if one_dev_combine_score > max_dev_score:
                    max_dev_score = one_dev_combine_score
                    torch.save({'model':model.state_dict()}, test_output_dir + best_save_name)
                    print("*** Found best model. Saving weights: {} ***".format(best_save_name[1:]))
                print ('----------------------------------------------------------------')

            model.train()

print(f"Training finished in {time.time() - start} seconds")    
