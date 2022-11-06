import pandas as pd
import json

def prepare_image_data(landmark2images, split):
    processed_data = json.load(open(f'./{split}/processed_wikilandmark_{split}_data.jsonl'))
    landmark_ids_in_split = [item['landmark_id'] for item in processed_data]
    landmark_images = [landmark2images[l_id] for l_id in landmark_ids_in_split]
    return landmark_images

if __name__ == '__main__':
    landmark2images = json.load(open('./raw_data/landmark2images.json'))
    splits = ('train', 'val', 'test')

    for split in splits:
        landmark_images = prepare_image_data(landmark2images, split)
        with open(f'./{split}/{split}_image_data.json', 'w') as fp:
            json.dump(landmark_images, fp, indent=4)
