/root/miniconda3/envs/ByProt/bin/pip install pandas Bio
/root/miniconda3/envs/ByProt/bin/pip install -e .
cp /hanchenchen/torch  /root/.cache/ -r  
env_name=ByProt 
# conda activate ${env_name} 
cd  /hanchenchen/byprot1/ByProt 
exp=fixedbb/lm_design_ours_650m 
dataset=cath_4.2
name=fixedbb/${dataset}/0915_lm_design_ours_650m_protmpnn_mlm_t33_numgpu_8
CUDA_LAUNCH_BLOCKING=1 /root/miniconda3/envs/ByProt/bin/python ./train.py \
name=${name} experiment=${exp} datamodule=${dataset}  \
logger=tensorboard trainer=ddp_fp16
