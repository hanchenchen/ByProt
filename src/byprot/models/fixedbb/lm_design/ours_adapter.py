from dataclasses import dataclass, field
from typing import List

import torch
from byprot.models import register_model
from byprot.models.fixedbb import FixedBackboneDesignEncoderDecoder
from byprot.models.fixedbb.generator import sample_from_categorical
from byprot.models.fixedbb.protein_mpnn_cmlm.protein_mpnn import (
    ProteinMPNNCMLM, ProteinMPNNConfig)

from .modules.ours_adapter import ESM2FoldseekWithStructuralAdatper


@dataclass
class ESM2AdapterConfig:
    encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [32, ])
    separate_loss: bool = True
    name: str = 'esm2_t33_650M_UR50D'
    dropout: float = 0.1
    # ensemble_logits: bool = False

t33_vocab = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
struc_vocab = "pynwrqhgdlvtmfsaeikc#"
standard_toks = []
for aa_token in seq_vocab:
    for struc_token in struc_vocab:
        standard_toks.append(aa_token+struc_token)
append_toks=["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]

aa_struc_vocab = append_toks + standard_toks
seq_vocab = append_toks + list(seq_vocab)
struc_vocab = append_toks + list(struc_vocab)

id_map = [t33_vocab.index(i) for i in seq_vocab if i!='#']

def merge_aa_struc(aa_tokens, struc_tokens):
    B, N = aa_tokens.shape
    aa_struc_tokens = torch.zeros_like(aa_tokens)
    for i in range(B):
        for j in range(N):
            aa_token = seq_vocab[aa_tokens[i][j]]
            struc_token = struc_vocab[struc_tokens[i][j]]
            if struc_token in append_toks:
                aa_struc_tokens[i][j] = aa_struc_vocab.index(struc_token)
            elif aa_token in append_toks:
                aa_token = '#'
                aa_struc_token = aa_token+struc_token
                aa_struc_tokens[i][j] = aa_struc_vocab.index(aa_struc_token)
            else:
                aa_struc_token = aa_token+struc_token
                aa_struc_tokens[i][j] = aa_struc_vocab.index(aa_struc_token)
    return aa_struc_tokens

def merge_aa_struc_func(aa_tokens, struc_tokens):
    B, N = aa_tokens.shape
    aa_tokens = aa_tokens-5
    aa_tokens = aa_tokens.masked_fill(aa_tokens<0, 20)

    struc_tokens_ = struc_tokens-5
    aa_struc_tokens = aa_tokens*21+struc_tokens_+5
    aa_struc_tokens[struc_tokens<5] = struc_tokens[struc_tokens<5]
    return aa_struc_tokens


@register_model('ours_adapter')
class ESM2Adapter(FixedBackboneDesignEncoderDecoder):
    _default_cfg = ESM2AdapterConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.encoder = ProteinMPNNCMLM(self.cfg.encoder)
        self.decoder = ESM2FoldseekWithStructuralAdatper.from_pretrained(args=self.cfg, name=self.cfg.name)

        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx

    def forward(self, batch, **kwargs):        
        # for k,v in batch.items():
        #     if isinstance(v, (torch.Tensor, )):
        #         print(k, v.shape)
        encoder_logits, encoder_out = self.encoder(batch, return_feats=True, **kwargs)
        encoder_logits = encoder_logits[:, :, id_map]

        encoder_out['feats'] = encoder_out['feats'].detach()

        init_pred = encoder_logits.argmax(-1)
        # assert (merge_aa_struc(init_pred, batch['struc_tokens']) == merge_aa_struc_func(init_pred, batch['struc_tokens'])).all(), (
        #     merge_aa_struc(init_pred, batch['struc_tokens']),
        #     merge_aa_struc_func(init_pred, batch['struc_tokens']),
        # )
        init_pred = merge_aa_struc_func(init_pred, batch['struc_tokens'])
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])

        esm_logits = self.decoder(
            tokens=init_pred,
            encoder_out=encoder_out,
        )['logits']
        if not getattr(self.cfg, 'separate_loss', False):
            logits = encoder_logits + esm_logits
            return logits, encoder_logits
        else:
            return esm_logits, encoder_logits

    def forward_encoder(self, batch):
        encoder_logits, encoder_out = self.encoder(batch, return_feats=True)
        encoder_logits = encoder_logits[:, :, id_map]

        init_pred = encoder_logits.argmax(-1)
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])

        encoder_out['logits'] = encoder_logits
        encoder_out['init_pred'] = init_pred
        encoder_out['coord_mask'] = batch['coord_mask']
        return encoder_out

    def forward_decoder(self, batch, prev_decoder_out, encoder_out, need_attn_weights=False):
        output_tokens = prev_decoder_out['output_tokens']
        output_scores = prev_decoder_out['output_scores']
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        # output_masks = output_tokens.eq(self.mask_idx)  # & coord_mask
        output_masks = output_tokens.ne(self.padding_idx)  # & coord_mask

        output_tokens_w_foldseek = merge_aa_struc_func(output_tokens, batch['struc_tokens'])
        esm_out = self.decoder(
            tokens=output_tokens_w_foldseek,
            encoder_out=encoder_out,
            need_head_weights=need_attn_weights
        )
        esm_logits = esm_out['logits']
        attentions = esm_out['attentions'] if need_attn_weights else None

        if not getattr(self.cfg, 'separate_loss', False):
            logits = esm_logits + encoder_out['logits']
        else:
            logits = esm_logits  # + encoder_out['logits']

        _tokens, _scores = sample_from_categorical(logits, temperature=temperature)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions, # [B, L, H, T, T]
            step=step + 1,
            max_step=max_step,
            history=history,
        )

    def initialize_output_tokens(self, batch, encoder_out):
        mask = encoder_out.get('coord_mask', None)

        prev_tokens = batch['prev_tokens']
        prev_token_mask = batch['prev_token_mask']
        # lengths = prev_tokens.ne(self.padding_idx).sum(1)

        # initial_output_tokens = torch.full_like(prev_tokens, self.padding_idx)
        # initial_output_tokens.masked_fill_(new_arange(prev_tokens) < lengths[:, None], self.mask_idx)
        # initial_output_tokens[:, 0] = self.cls_idx
        # initial_output_tokens.scatter_(1, lengths[:, None] - 1, self.eos_idx)

        # initial_output_tokens = encoder_out['init_pred'].clone()
        initial_output_tokens = torch.where(
            prev_token_mask, encoder_out['init_pred'], prev_tokens)
        initial_output_scores = torch.zeros(
            *initial_output_tokens.size(), device=initial_output_tokens.device
        )

        return initial_output_tokens, initial_output_scores
