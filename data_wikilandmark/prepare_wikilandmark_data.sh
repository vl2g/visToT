python3 data_processing.py\
    --special_token_path ./wikilandmark_col_header_vocab.txt\
    --special_token_min_cnt 10\
    --raw_data_path ./train/processed_wikilandmark_train_data.jsonl\
    --file_head_name ./train/wikilandmark_train\
    --dataset_mode train

python3 data_processing.py\
    --special_token_path ./wikilandmark_col_header_vocab.txt\
    --special_token_min_cnt 10\
    --raw_data_path ./val/processed_wikilandmark_val_data.jsonl\
    --file_head_name ./val/wikilandmark_val\
    --dataset_mode val

python3 data_processing.py\
    --special_token_path ./wikilandmark_col_header_vocab.txt\
    --special_token_min_cnt 10\
    --raw_data_path ./test/processed_wikilandmark_test_data.jsonl\
    --file_head_name ./test/wikilandmark_test\
    --dataset_mode test
