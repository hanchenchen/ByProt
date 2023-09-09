# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

import esm
from esm.modules import (
    TransformerLayer,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
    ESM1LayerNorm,
    FeedForwardNetwork,
    NormalizedResidualBlock,
    gelu,
)
from esm.multihead_attention import MultiheadAttention
from byprot.utils.config import compose_config as Cfg, merge_config
from transformers import EsmTokenizer, EsmForMaskedLM, EsmModel


tokenizer = EsmTokenizer.from_pretrained("/hanchenchen/BACKUP_20230628/EVE/SS/ss_650m_0606")

vocab = tokenizer.get_vocab()
t33_vocab = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
taa_vocab = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
aa_id_map = [seq_vocab.index(i) for i in taa_vocab]
id_map = {t33_vocab.index(i):(vocab[i+'#'] if i in taa_vocab else (vocab[i] if i in vocab else vocab['<unk>'])) for i in t33_vocab}
print(id_map)
# seq_vocab = [i if i in seq_vocab else 'A' for i in t33_vocab]
struc_vocab = "pynwrqhgdlvtmfsaeikc#"
def sum_seq_prob(pred_prob):
    """
    
    Args:
        pred_prob: [N, vocab_size]

    Returns:
        seq_prob:   [N, 20]
    """
    
    B, N, C = pred_prob.shape
    pred_prob = pred_prob[:, :, 5:425]
    pred_prob = pred_prob.reshape(B, N, 20, 21).sum(dim=-1)
    pred_prob = pred_prob[:, :, aa_id_map]
    left_pad = pred_prob[:, :, :4]
    right_pad = torch.ones(B, N, 9).to(pred_prob)*(-1e4)
    pred_prob = torch.cat((left_pad, pred_prob, right_pad), dim=2)
    return pred_prob


class ESM2WithStructuralAdatper(nn.Module):
    @classmethod
    def from_pretrained(cls, args, override_args=None, name='esm2_t33_650M_UR50D'):
        import esm
        pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)
        print(pretrained_model.state_dict().keys(), alphabet.all_toks)

        pretrained_args = Cfg(
            num_layers=pretrained_model.num_layers, 
            embed_dim=pretrained_model.embed_dim, 
            attention_heads=pretrained_model.attention_heads, 
            token_dropout=pretrained_model.token_dropout, 
        )
        args = merge_config(pretrained_args, args)
        # args.adapter_layer_indices = getattr(args, 'adapter_layer_indices', [6, 20, 32])

        args.adapter_layer_indices = [-1]
        args.adapter_layer_indices = list(
            map(lambda x: (args.num_layers + x) % args.num_layers, 
                args.adapter_layer_indices)
        )

        model = cls(args, deepcopy(alphabet)) 

        ckpt_path = "/hanchenchen/BACKUP_20230628/EVE/SS/650M/esm2_t33_650M_UR50D_foldseek_plddt70_iter2900448.pt"
        model_data = torch.load(ckpt_path, map_location="cpu")

        # Convert the keys
        weights = {k.replace("esm.encoder.layer", "layers"): v for k, v in model_data["model"].items()}
        weights = {k.replace("attention.self", "self_attn"): v for k, v in weights.items()}
        weights = {k.replace("key", "k_proj"): v for k, v in weights.items()}
        weights = {k.replace("query", "q_proj"): v for k, v in weights.items()}
        weights = {k.replace("value", "v_proj"): v for k, v in weights.items()}
        weights = {k.replace("attention.output.dense", "self_attn.out_proj"): v for k, v in weights.items()}
        weights = {k.replace("attention.LayerNorm", "self_attn_layer_norm"): v for k, v in weights.items()}
        weights = {k.replace("intermediate.dense", "fc1"): v for k, v in weights.items()}
        weights = {k.replace("output.dense", "fc2"): v for k, v in weights.items()}
        weights = {k.replace("LayerNorm", "final_layer_norm"): v for k, v in weights.items()}
        weights = {k.replace("esm.embeddings.word_embeddings", "embed_tokens"): v for k, v in weights.items()}
        weights = {k.replace("rotary_embeddings", "rot_emb"): v for k, v in weights.items()}
        weights = {k.replace("embeddings.LayerNorm", "embed_layer_norm"): v for k, v in weights.items()}
        weights = {k.replace("esm.encoder.", ""): v for k, v in weights.items()}
        weights = {k.replace("lm_head.decoder.weight", "lm_head.weight"): v for k, v in weights.items()}


        model.load_state_dict(weights, strict=False)        

        del pretrained_model

        # freeze pretrained parameters
        for pname, param in model.named_parameters():
            if 'adapter' not in pname:
                print('Fix', pname)
                param.requires_grad = False
            else:
                print('Train', pname)

        return model 

    def __init__(
        self,
        args,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        # num_layers: int = 33,
        # embed_dim: int = 1280,
        # attention_heads: int = 20,
        # token_dropout: bool = True,
    ):
        super().__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim
        self.attention_heads = args.attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = 446 # len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = args.token_dropout

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                self._init_layer(_)
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def _init_layer(self, layer_idx):
        if layer_idx in self.args.adapter_layer_indices:
            layer = TransforerLayerWithStructralAdapter(
                self.embed_dim,
                4 * self.embed_dim,
                self.attention_heads,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True,
                encoder_embed_dim=self.args.encoder.d_model,
                dropout=self.args.dropout
            )
        else:
            layer = TransformerLayer(
                self.embed_dim,
                4 * self.embed_dim,
                self.attention_heads,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True,
            )
        return layer

    def forward_layers(self, x, encoder_out, padding_mask, repr_layers=[], hidden_representations=[], need_head_weights=False, attn_weights=[]):
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.args.adapter_layer_indices:
                x, attn = layer(
                    x, encoder_out, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
                )
            else:
                x, attn = layer(
                    x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
                )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        return x, hidden_representations, attn_weights, layer_idx

    def forward(self, tokens, encoder_out, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

         
        _tokens = torch.ones_like(tokens)
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                _tokens[i][j] = id_map[tokens[i][j].item()]
        _tokens = _tokens.to(tokens)
        x = self.embed_scale * self.embed_tokens(_tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        # for layer_idx, layer in enumerate(self.layers):
        #     x, attn = layer(
        #         x,
        #         self_attn_padding_mask=padding_mask,
        #         need_head_weights=need_head_weights,
        #     )
        #     if (layer_idx + 1) in repr_layers:
        #         hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        #     if need_head_weights:
        #         # (H, B, T, T) => (B, H, T, T)
        #         attn_weights.append(attn.transpose(1, 0))

        x, hidden_representations, attn_weights, layer_idx = self.forward_layers(
            x, encoder_out, padding_mask, 
            repr_layers=repr_layers, 
            hidden_representations=hidden_representations,
            need_head_weights=need_head_weights,
            attn_weights=attn_weights if need_head_weights else None
        )


        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)
        x = sum_seq_prob(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]


class TransforerLayerWithStructralAdapter(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        encoder_embed_dim,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings

        self.encoder_embed_dim = encoder_embed_dim
        self.dropout = dropout
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)


    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

        # structural adapter
        self.structural_adapter_attn = NormalizedResidualBlock(
            layer=MultiheadAttention(
                self.embed_dim,
                self.attention_heads,
                kdim=self.encoder_embed_dim,
                vdim=self.encoder_embed_dim,
                add_bias_kv=add_bias_kv,
                add_zero_attn=False,
                use_rotary_embeddings=True,
            ),
            embedding_dim=self.embed_dim,
            dropout=self.dropout
        )
        self.structural_adapter_ffn = NormalizedResidualBlock(
            layer=FeedForwardNetwork(
                self.embed_dim,
                self.embed_dim // 2, # NOTE: bottleneck FFN is important
                # self.ffn_embed_dim,
                activation_dropout=self.dropout
            ),
            embedding_dim=self.embed_dim,
            dropout=self.dropout
        )

    def forward(
        self, x, encoder_out, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        # x = self.forward_adapter(x, encoder_out, attn_mask=self_attn_mask, attn_padding_mask=self_attn_padding_mask)

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        x = x + self.forward_adapter(x, encoder_out, attn_mask=self_attn_mask, attn_padding_mask=self_attn_padding_mask)
        return x, attn

    def forward_adapter(self, x, encoder_out, attn_mask, attn_padding_mask):
        encoder_feats = encoder_out['feats']
        encoder_feats = encoder_feats.transpose(0, 1)

        x = self.structural_adapter_attn(
            x, 
            key=encoder_feats,
            value=encoder_feats,
            key_padding_mask=attn_padding_mask,
            attn_mask=attn_mask,
            need_weights=False
        )[0]

        x = self.structural_adapter_ffn(x)
        # x = x.transpose(0, 1)
        return x