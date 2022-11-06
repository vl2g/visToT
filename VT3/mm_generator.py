import torch
from torch import nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from model.modeling_bart import ModifiedVLBart as model

class MMGenerator(nn.Module):
    def __init__(
            self,
            model_name,
            tokenizer,
            max_decode_len,
            dropout,
            feat_dim,
            use_vis_layer_norm,
            individual_vis_layer_norm,
            share_vis_lang_layer_norm,
            ):
        super().__init__()
        self.tokenizer = tokenizer # tokenizer with extended vocabulary
        self.max_decode_len = max_decode_len

        print ('Initializing model...')
        bart_config = BartConfig.from_pretrained(model_name)
        bart_config.__dict__["dropout"] = dropout
        bart_config.feat_dim = feat_dim
        bart_config.use_vis_layer_norm = use_vis_layer_norm
        bart_config.individual_vis_layer_norm = individual_vis_layer_norm
        bart_config.share_vis_lang_layer_norm = share_vis_lang_layer_norm
        bart_config.dropout_rate = dropout
        bart_config.attention_dropout = dropout
        bart_config.activation_dropout = dropout
        bart_config.use_vis_order_embedding = False
        bart_config.n_images = 1

        self.model = model.from_pretrained(model_name, config=bart_config)
        print ('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.vocab_size = len(self.tokenizer)
        self.logsftmax = nn.LogSoftmax(dim=-1)
        self.padding_idx = self.tokenizer.pad_token_id

    def forward(self, src_input, src_mask, src_vis_inputs, tgt_input, tgt_output):
        src_mask = src_mask.type(src_input.type())
        outputs = self.model(
                input_ids=src_input,
                attention_mask=src_mask,
                vis_inputs=src_vis_inputs,
                decoder_input_ids=tgt_input,
                labels=tgt_output,
                reduce_loss=True,
            )
        loss = outputs['loss']
        return loss

    def generate(self, src_input, src_mask, src_vis_inputs):
        result_list = []
        outputs = self.model.generate(
                input_ids=src_input,
                attention_mask=src_mask,
                vis_inputs=src_vis_inputs,
                max_length=self.max_decode_len,
            )
        for predicted_ids in outputs:
            one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
            result_list.append(one_result)
        return result_list