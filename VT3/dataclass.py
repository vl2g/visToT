from re import I
import sys
import os
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
from transformers import BartTokenizer, BartTokenizerFast
# from data_utils import load_ordered_cell_list

UNSEEN_SLOT_KEY, EOS, SEP = '__None__', '__EOS__', '__SEP__'

def shift_tokens_right(input_ids, pad_token_id):
    """
    Borrowed old implementation of modeling_bart.shift_tokens_right (v3.1.0)
    Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

class Data:
    def __init__(self, train_data_dict, dev_data_dict, image_feat_path, max_table_len, max_tgt_len, 
        model_name, special_token_path, min_slot_key_cnt, pretrain_mode=False):

        self.pretrain_mode = pretrain_mode
        self.image_feat_path = image_feat_path
        self.max_table_len, self.max_tgt_len = \
        max_table_len, max_tgt_len
        self.special_token_list, self.special_token_dict = [], {}
        with open(special_token_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_special_token = l.strip('\n').split()[0]
                cnt = int(l.strip('\n').split()[1])
                if cnt >= min_slot_key_cnt: # if slot key's freq count is above threshold then include in new vocabulary
                    self.special_token_list.append(one_special_token)
                    self.special_token_dict[one_special_token] = 1
                else:
                    pass
        print ('Number of Special Token is %d' % len(self.special_token_list)) # num of unique special tokens/slot keys

        self.model_name = model_name
        self.tokenizer = BartTokenizerFast.from_pretrained(model_name)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decode_tokenizer = BartTokenizer.from_pretrained(model_name)

        print ('original vocabulary Size %d' % len(self.tokenizer))
        self.tokenizer.add_tokens(self.special_token_list) # Add new special tokens/ slot keys etc to tokenizer vocab
        self.decode_tokenizer.add_tokens(self.special_token_list)
        print ('vocabulary size after extension is %d' % len(self.tokenizer))

        self.sep_idx = self.tokenizer.convert_tokens_to_ids([SEP])[0]
        self.eos_idx = self.tokenizer.convert_tokens_to_ids([EOS])[0]
        print (self.eos_idx)

        print ('Start loading training data...')
        self.train_tabel_id_list, self.train_tgt_id_list, self.train_image_data = self.load_data(train_data_dict)

        # if use_RL:
        #     print ('Loading Training Ordered Cell list...')
        #     train_processed_file_path = train_data_dict['processed_file_path']
        #     self.train_ordered_cell_list = load_ordered_cell_list(train_processed_file_path, self.special_token_dict)
        # else:
        #     self.train_ordered_cell_list = [[] for _ in range(len(self.train_tabel_id_list))]
        print ('Training data loaded.')

        print ('Start loading validation data...')
        self.dev_tabel_id_list, self.dev_tgt_id_list, self.dev_image_data = self.load_data(dev_data_dict)

        # TODO: check this and remove
        # if use_RL:
        #     print ('Loading Validation Ordered Cell list...')
        #     dev_processed_file_path = dev_data_dict['processed_file_path']
        #     self.dev_ordered_cell_list = load_ordered_cell_list(dev_processed_file_path, self.special_token_dict)
        # else:
        #     self.dev_ordered_cell_list = [[] for _ in range(len(self.dev_tabel_id_list))]
        print ('Validation data loaded.')

        self.train_num, self.dev_num = len(self.train_tabel_id_list), len(self.dev_tabel_id_list)
        self.train_idx_list = [i for i in range(self.train_num)]
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.dev_current_idx = 0

        self.train_num, self.dev_num = 0, 0 # get actual lengths, ie total number of (landmark, image) pairs
        for images in self.train_image_data:
            self.train_num += len(images)
        for images in self.dev_image_data:
            self.dev_num += len(images)
        print ('train number is %d, dev number is %d' % (self.train_num, self.dev_num))
        self.dev_all_item_list = []
        for i in range(len(self.dev_tabel_id_list)):
            for j in range(len(self.dev_image_data[i])):
                self.dev_all_item_list.append((i, j)) # an index for every (table, image) sample in dev
        assert len(self.dev_all_item_list) == self.dev_num


    def load_one_text_id(self, text, max_len):
        text_id_list = self.tokenizer.encode(text, max_length=512, truncation=True, add_special_tokens=False)[:max_len]
        return text_id_list

    def load_text_id_list(self, path, max_len):
        text_list = []
        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                text_list.append(l.strip('\n'))

        p = progressbar.ProgressBar(len(text_list))
        p.start()
        res_id_list = []
        idx = 0
        for text in text_list:
            p.update(idx + 1)
            one_id_list = self.load_one_text_id(text, max_len) # Get BART tokenizer encoded token-ids [:max_len]
            res_id_list.append(one_id_list)
            idx += 1
        p.finish()
        return res_id_list

    def load_data(self, data_dict):
        table_text_path = data_dict['table_text_path']
        print ('Loading Table Data...')
        tabel_id_list = self.load_text_id_list(table_text_path, self.max_table_len)
        tabel_id_list = [[self.sep_idx] + one_id_list for one_id_list in tabel_id_list] # septokenid + table token ids

        # print ('Loading Content Data...')
        # content_text_path = data_dict['content_text_path']
        # content_id_list = self.load_text_id_list(content_text_path, self.max_content_plan_len)
        # content_id_list = [[self.sep_idx] + one_id_list for one_id_list in content_id_list]
        # assert len(tabel_id_list) == len(content_id_list)

        print ('Loading Reference Data...')
        reference_sentence_path = data_dict['reference_sentence_path']
        tgt_id_list = self.load_text_id_list(reference_sentence_path, self.max_tgt_len)
        assert len(tabel_id_list) == len(tgt_id_list)

        tgt_id_list = [[self.bos_token_id] + item + [self.eos_token_id] for item in tgt_id_list]

        # id_content_dict = {} # idx -> content text
        # with open(content_text_path, 'r', encoding = 'utf8') as i:
        #     lines = i.readlines()
        #     idx = 0
        #     for l in lines:
        #         one_content_text = l.strip('\n')
        #         id_content_dict[idx] = one_content_text
        #         idx += 1

        # reference_text_list = []
        # with open(reference_sentence_path, 'r', encoding = 'utf8') as i:
        #     lines = i.readlines()
        #     for l in lines:
        #         one_text = l.strip('\n')
        #         reference_text_list.append(one_text)

        # reference_content_plan_list = [] # list of content texts
        # with open(content_text_path, 'r', encoding = 'utf8') as i:
        #     lines = i.readlines()
        #     for l in lines:
        #         reference_content_plan_list.append(l.strip('\n'))

        print("Loading Image ID Data...")
        image_data = json.load(open(data_dict['image_data_path'])) # [[img_id1, ...], ...] list of lists, len = num of landmarks
        
        return tabel_id_list, tgt_id_list, image_data

    def get_image_feats(self, image_id):
        # print(image_id)
        image_feats = np.load(os.path.join(self.image_feat_path, image_id + '.npy'))
        image_feats = torch.from_numpy(image_feats)
        return image_feats
    
    def process_vis_tensor(self, batch_image_id_list):
        batch_image_feats = [self.get_image_feats(image_id) for image_id in batch_image_id_list]
        batch_image_feats = torch.stack(batch_image_feats)
        return batch_image_feats

    def process_source_tensor(self, batch_src_id_list):
        batch_src_tensor_list = [torch.LongTensor(item) for item in batch_src_id_list] # convert list to LongTensor
        batch_src_tensor = rnn.pad_sequence(batch_src_tensor_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        if self.pretrain_mode:
            batch_src_tensor = torch.ones_like(batch_src_tensor)

        # ---- compute src mask ---- #
        batch_src_mask = torch.ones_like(batch_src_tensor)
        batch_src_mask = batch_src_mask.masked_fill(batch_src_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_src_tensor, batch_src_mask

    def process_decoder_tensor(self, batch_tgt_id_list):
        batch_tgt_tensor = [torch.LongTensor(item) for item in batch_tgt_id_list] # convert list to LongTensor
        batch_tgt_tensor = rnn.pad_sequence(batch_tgt_tensor, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_labels = batch_tgt_tensor
        batch_input = shift_tokens_right(batch_labels, self.tokenizer.pad_token_id) 
        batch_labels[batch_labels[:, :] == self.tokenizer.pad_token_id] = -100 # -100 label indicates to decoder not to generate a token for this position
        return batch_input, batch_labels 

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        # batch_content_set = set()
        # for one_idx in batch_idx_list:
        #     batch_content_set.add(self.train_id_content_dict[one_idx])

        # # we ensure the same batch does not contain repeated content plan # Q: Why?
        # while len(batch_content_set) < batch_size:
        #     batch_idx_list = random.sample(self.train_idx_list, batch_size) # sample idxes of unique landmarks (not landmark & image)
        #     batch_content_set = set()
        #     for one_idx in batch_idx_list:
        #         batch_content_set.add(self.train_id_content_dict[one_idx])

        batch_src_id_list, batch_image_id_list, batch_tgt_id_list = [], [], []
        # data used for RL training
        batch_ordered_cell_list, batch_reference_text_list, batch_content_plan_list = [], [], []
        for idx in batch_idx_list:
            one_table_id_list = self.train_tabel_id_list[idx] # bart tokenized ids for table
            one_image_id = random.choice(self.train_image_data[idx]) # sample random image for a landmark/table from its set of images
            # one_content_id_list = self.train_content_id_list[idx]
            one_tgt_id_list = self.train_tgt_id_list[idx]

            # batch_table_id_list.append(one_table_id_list)
            # batch_content_id_list.append(one_content_id_list)
            # if self.use_CP:
            #     batch_src_id_list.append(one_table_id_list + one_content_id_list)
            # else:
            batch_src_id_list.append(one_table_id_list)
            batch_image_id_list.append(one_image_id)
            batch_tgt_id_list.append(one_tgt_id_list)

            # one_ordered_cell_list = self.train_ordered_cell_list[idx]
            # batch_ordered_cell_list.append(one_ordered_cell_list)
            # one_reference_text = self.train_reference_text_list[idx]
            # batch_reference_text_list.append(one_reference_text)
            # one_content_plan = self.train_reference_content_plan_list[idx]
            # batch_content_plan_list.append(one_content_plan)

        # batch_table_tensor, batch_table_mask = self.process_source_tensor(batch_table_id_list) # adds input ids padding and also creates attention mask
        # batch_content_tensor, batch_content_mask = self.process_source_tensor(batch_content_id_list)
        batch_src_tensor, batch_src_mask = self.process_source_tensor(batch_src_id_list)
        batch_src_vis_feats_tensor = self.process_vis_tensor(batch_image_id_list)
        batch_tgt_in_tensor, batch_tgt_out_tensor = self.process_decoder_tensor(batch_tgt_id_list)
        return (batch_src_tensor, batch_src_mask), (batch_src_vis_feats_tensor,), \
        (batch_tgt_in_tensor, batch_tgt_out_tensor)

    def get_next_dev_batch(self, idx, batch_size):
        batch_src_id_list, batch_image_id_list, batch_tgt_id_list = [], [], []
        # batch_ordered_cell_list, batch_reference_text_list, batch_content_plan_list = [], [], []

        start_idx = idx * batch_size
        end_idx = (idx+1) * batch_size
        end_idx = min(end_idx, self.dev_num)
        for item_idx in range(start_idx, end_idx):
            curr_idx, image_idx = self.dev_all_item_list[item_idx] # (curr_idx = table idx, image idx)
            one_table_id_list = self.dev_tabel_id_list[curr_idx]
            # one_content_id_list = self.dev_content_id_list[curr_idx]
            one_tgt_id_list = self.dev_tgt_id_list[curr_idx]
            one_image_id = self.dev_image_data[curr_idx][image_idx]

            # batch_table_id_list.append(one_table_id_list)
            # batch_content_id_list.append(one_content_id_list)
            # if self.use_CP:
            #     batch_src_id_list.append(one_table_id_list + one_content_id_list)
            # else:
            batch_src_id_list.append(one_table_id_list)
            batch_image_id_list.append(one_image_id)
            batch_tgt_id_list.append(one_tgt_id_list)

            # one_ordered_cell_list = self.dev_ordered_cell_list[curr_idx]
            # batch_ordered_cell_list.append(one_ordered_cell_list)
            # one_reference_text = self.dev_reference_text_list[curr_idx]
            # batch_reference_text_list.append(one_reference_text)
            # one_content_plan = self.dev_reference_content_plan_list[curr_idx]
            # batch_content_plan_list.append(one_content_plan)

        # batch_table_tensor, batch_table_mask = self.process_source_tensor(batch_table_id_list)
        # batch_content_tensor, batch_content_mask = self.process_source_tensor(batch_content_id_list)
        batch_src_tensor, batch_src_mask = self.process_source_tensor(batch_src_id_list)
        batch_src_vis_feats_tensor = self.process_vis_tensor(batch_image_id_list)
        batch_tgt_in_tensor, batch_tgt_out_tensor = self.process_decoder_tensor(batch_tgt_id_list)
        
        return (batch_src_tensor, batch_src_mask), (batch_src_vis_feats_tensor,), \
        (batch_tgt_in_tensor, batch_tgt_out_tensor)

