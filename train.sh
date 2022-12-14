CUDA_VISIBLE_DEVICES=0,1 python3 ./VT3/train.py\
    --train_table_text_path ./data_wikilandmark/train/wikilandmark_train_table.txt\
    --train_reference_sentence_path ./data_wikilandmark/train/wikilandmark_train_reference.txt\
    --train_image_data_path ./data_wikilandmark/train/train_image_data.json\
    --dev_table_text_path ./data_wikilandmark/val/wikilandmark_val_table.txt\
    --dev_reference_sentence_path ./data_wikilandmark/val/wikilandmark_val_reference.txt\
    --dev_image_data_path ./data_wikilandmark/val/val_image_data.json\
    --dev_reference_path ./data_wikilandmark/val/processed_wikilandmark_val_data.jsonl\
    --image_feat_path ./data_wikilandmark/image_features\
    --special_token_path ./data_wikilandmark/wikilandmark_col_header_vocab.txt\
    --output_path ./VT3/output/finetune/\
    --pretrained_ckpt_path ./VT3/output/pretrain/best-checkpoint.ckpt\
    --eval_every 2000\
    --feat_dim 1536\
    --max_table_len 256\
    --max_tgt_len 110\
    --batch_size 100\
    --gradient_accumulation_steps 2