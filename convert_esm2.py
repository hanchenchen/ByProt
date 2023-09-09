import torch
import esm
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


pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub('esm2_t33_650M_UR50D')
pretrained_model.load_state_dict(weights, strict=True)        
