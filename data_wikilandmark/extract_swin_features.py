import torch
from transformers import AutoFeatureExtractor, SwinForImageClassification
import numpy as np
from PIL import Image, ImageFile
import argparse
import os
import glob
from tqdm import tqdm
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FeatureExtractor:

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.model_name = self.args.model_name
        self.vit_feature_extractor = self._build_vit_feature_extractor()
        self.vit_model = self._build_vit_model()
        os.makedirs(self.args.output_dir, exist_ok=True)

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Extract image features using pretrained SWiN')
        parser.add_argument('--image_dir', help='image directory')
        parser.add_argument('--output_dir', help='output directory')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        parser.add_argument('--model_name', default='microsoft/swin-large-patch4-window12-384-in22k', help='name of pretrained classification ViT model')
        return parser

    def _build_vit_feature_extractor(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        return feature_extractor

    def _build_vit_model(self):
        model = SwinForImageClassification.from_pretrained(self.model_name)

        model.to("cuda")
        model.eval()
        return model

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        file_base_name = file_base_name + '.npy'

        np.save(os.path.join(self.args.output_dir, file_base_name), feature)

    def get_vit_features(self, image_paths):
        images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images.append(image)
        
        inputs = self.vit_feature_extractor(images=images, return_tensors="pt") 
        inputs.to('cuda')

        with torch.no_grad():
            outputs = self.vit_model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.hidden_states[-1]
        
        feat_list = []
        num_images = last_hidden_states.size()[0]
        for i in range(num_images):
            feat_list.append(last_hidden_states[i].cpu().numpy())

        return feat_list

    def extract_features(self):
        image_dir = self.args.image_dir

        if os.path.isfile(image_dir):
            if not image_dir.split('.')[-1] == 'json':
                # single image feature extraction
                features = self.get_vit_features([image_dir])
                self._save_feature(image_dir, features[0])
            else:
                # json file of image paths
                files = json.load(open(image_dir))
                print("Number of input images: ", len(files))
                print("Batch size", self.args.batch_size)
                num_batches = round(len(files)/self.args.batch_size)
                for chunk in tqdm(self._chunks(files, self.args.batch_size), total=num_batches, desc="batches processed"):
                    features = self.get_vit_features(chunk)
                    for idx, file_name in enumerate(chunk):
                        self._save_feature(file_name, features[idx])
        else:
            files = glob.glob(os.path.join(image_dir, "*.*"))
            print("Number of input images: ", len(files))
            print("Batch size", self.args.batch_size)
            num_batches = round(len(files)/self.args.batch_size)
            for chunk in tqdm(self._chunks(files, self.args.batch_size), total=num_batches, desc="batches processed"):
                features = self.get_vit_features(chunk)
                for idx, file_name in enumerate(chunk):
                    self._save_feature(file_name, features[idx])

if __name__ == "__main__":

    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
