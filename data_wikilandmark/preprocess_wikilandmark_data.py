import json
import os
import csv
import argparse
import jsonlines

def read_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', default='./raw_data', type=str)
    return parser.parse_args()

def get_captions(captions_path):
    csvreader = csv.reader(open(captions_path, encoding='utf8'))
    header = next(csvreader)
    captions = {}
    for row in csvreader:
        # each row: [landmarkid, caption, imageid]
        landmark_id, caption, image_id = row[0], row[1], row[2]
        if landmark_id in captions:
            continue
        else:
            captions[landmark_id] = caption
    return captions


def get_infoboxes(infoboxes_path):
    infoboxes = {}
    infobox_files = os.listdir(infoboxes_path)
    for infobox_file in infobox_files:
        landmark_id = infobox_file.split('.')[0]
        infobox_data = read_json(os.path.join(infoboxes_path, infobox_file))
        infoboxes[landmark_id] = infobox_data
    return infoboxes
    

def get_data(ids, infoboxes, captions):
    data = []
    ids = list(map(str, ids))
    filtered_ids = set(ids) & set(infoboxes.keys()) & set(captions.keys())

    for landmark_id in filtered_ids:
        infobox = infoboxes[landmark_id]
        caption = captions[landmark_id]
        if len(caption) == 0 or len(infobox) == 0:
            continue
        data.append({"landmark_id": landmark_id, "infobox": infobox, "sentence": caption})
    return data

if __name__ == '__main__':
    args = parse_config()
    raw_data_path = args.raw_data_path
    
    data_path = os.path.join(raw_data_path, 'data.json')
    train_ids_path = os.path.join(raw_data_path, 'train_landmarks.json')
    val_ids_path = os.path.join(raw_data_path, 'val_landmarks.json')
    test_ids_path = os.path.join(raw_data_path, 'test_landmarks.json')

    data = json.load(open(data_path))
    captions = dict(zip(data['landmark_id'], data['caption']))
    infoboxes = dict(zip(data['landmark_id'], data['table']))
    train_ids = read_json(train_ids_path)
    val_ids = read_json(val_ids_path)
    test_ids = read_json(test_ids_path)
    train_data = get_data(train_ids, infoboxes, captions)
    val_data = get_data(val_ids, infoboxes, captions)
    test_data = get_data(test_ids, infoboxes, captions)

    os.makedirs('./train/', exist_ok=False)
    os.makedirs('./val/', exist_ok=False)
    os.makedirs('./test/', exist_ok=False)

    with jsonlines.open('./train/processed_wikilandmark_train_data.jsonl', mode='w') as f:
        f.write(train_data)
    
    with jsonlines.open('./val/processed_wikilandmark_val_data.jsonl', mode='w') as f:
        f.write(val_data)

    with jsonlines.open('./test/processed_wikilandmark_test_data.jsonl', mode='w') as f:
        f.write(test_data)