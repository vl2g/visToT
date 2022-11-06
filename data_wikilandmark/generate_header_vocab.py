import argparse
import os
import json
import pandas as pd

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./raw_data/data.json')
    parser.add_argument('--output_file', default='wikilandmark_col_header_vocab.txt', type=str)
    parser.add_argument('--train_split_file', type=str, default='./raw_data/train_landmarks.json')
    return parser.parse_args()

def add_vocab(table, header_vocab):
    for slot_key, slot_val in table.items():
        if not slot_val:
            continue
        formatted_slot_key = '__' + '_'.join(slot_key.split()).strip('_') + '__'
        if formatted_slot_key not in header_vocab:
            header_vocab[formatted_slot_key] = 1
        elif header_vocab[formatted_slot_key] <= 50000:
            header_vocab[formatted_slot_key] += 1
    return header_vocab

def get_landmark_ids_in_captions(captions_path):
    captions_data = pd.read_csv(captions_path)
    landmark_ids_in_captions = captions_data['landmark_id'].tolist()
    landmark_ids_in_captions = list(set(landmark_ids_in_captions))
    landmark_ids_in_captions = list(map(str, landmark_ids_in_captions))
    return landmark_ids_in_captions

def get_landmark_ids_in_infoboxes(infobox_files):
    landmark_ids_in_infoboxes = [filename.split('.')[0] for filename in infobox_files]
    return landmark_ids_in_infoboxes

if __name__ == '__main__':
    args = parse_config()

    all_data = json.load(open(args.data))
    infoboxes = all_data['table']

    if args.train_split_file:
        train_ids = json.load(open(args.train_split_file))
        infoboxes = []
        for (landmark_id, table) in zip(all_data['landmark_id'], all_data['table']):
            if landmark_id in train_ids:
                infoboxes.append(table)

    header_vocab = {}
    header_vocab['__EOS__'] = 50000
    header_vocab['__SEP__'] = 50000
    header_vocab['__None__'] = 50000
    
    for idx, table in enumerate(infoboxes):
        header_vocab = add_vocab(table, header_vocab)

    with open(args.output_file, 'w') as f:
        header_vocab = [(slot_value, slot_key) for slot_value, slot_key in header_vocab.items()]
        header_vocab.sort(key = lambda x: -1 * int(x[1]))
        for (slot_key, slot_value) in header_vocab:
            line = slot_key + ' ' + str(slot_value)
            f.write(line)
            f.write('\n')
