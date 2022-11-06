# VisToT

**Code for VisToT: Vision-Augmented Table-to-Text Generation (EMNLP 2022) paper.**

[Project Page](https://vl2g.github.io/projects/vistot) | [Paper](https://vl2g.github.io/projects/vistot/docs/VISTOT-EMNLP2022.pdf)

## Requirements
- Use **python >= 3.8.0**. Conda recommended: [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)
- Use Pytorch >=1.7.0 for CUDA 11.0 : [https://pytorch.org/get-started/previous-versions/#linux-and-windows-20](https://pytorch.org/get-started/previous-versions/#linux-and-windows-20)
- Other requirements are listed in `requirements.txt`

**Setup the environment**
```bash
# create a new conda environment
conda create -n vt3 python=3.8.13

# activate environment
conda activate vt3

# install pytorch
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# install other dependencies
pip install -r requirements.txt
```

## Dataset Preparation
Download the dataset [here](). Move the downloaded content to `./data_wikilandmark` directory and unzip the files.

```bash
# go to directory
cd data_wikilandmark
```

**Image Feature Extraction**
```bash
python extract_swin_features.py --input_dir ./images/ --output_dir ./image_features/
```

**Prepare dataset**
```bash
./generate_dataset_files.sh
```

## Training & Evaluation
The scripts for training and evaluation are in `./VT3` directory.

Download trained checkpoint for VT3: [Google Drive Link](https://drive.google.com/drive/folders/1mSjb2DHEL5bU5r4oeN3I8yG96zKJSnnH?usp=sharing)

**Training**
```bash
# pretrain VT3 model and save checkpoint
./pretrain.sh

# finetune VT3 model on wikilandmarks dataset using saved checkpoint
./train.sh
```

**Evaluation**
```bash
# perform inference on test data
./perform_inference.sh
```

## Cite
If you find this work useful for your research, please consider citing.
<pre><tt>@inproceedings{vistot2022emnlp,
  author    = "Gatti, Prajwal and 
              Mishra, Anand and
              Gupta, Manish and
              Das Gupta, Mithun"
  title     = "VisToT: Vision-Augmented Table-to-Text Generation",
  booktitle = "EMNLP",
  year      = "2022",
}</tt></pre>

## Acknowledgements
This implementation is based on the code provided by [https://github.com/yxuansu/PlanGen](https://github.com/yxuansu/PlanGen).
<br>Code provided by [https://github.com/j-min/VL-T5/blob/main/VL-T5/src/modeling_bart.py](https://github.com/j-min/VL-T5/blob/main/VL-T5/src/modeling_bart.py) helped in implementing VT3 transformer.
<br>Swin Transformer used for feature extraction was provided by [https://huggingface.co/docs/transformers/model_doc/swin](https://huggingface.co/docs/transformers/model_doc/swin).

## License  
This code is released under the [MIT license](./LICENSE.md).
