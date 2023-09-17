/root/miniconda3/envs/ByProt/bin/pip install pandas Bio
/root/miniconda3/envs/ByProt/bin/pip install -e .

cp /hanchenchen/torch  /root/.cache/ -r  
env_name=ByProt 
conda activate ${env_name} 
cd  /hanchenchen/byprot1/ByProt
exp=fixedbb/lm_design_esm1b_650m 
dataset=cath_4.2
name=fixedbb/${dataset}/0912_lm_design_esm1b_650m 
CUDA_LAUNCH_BLOCKING=1 /root/miniconda3/envs/ByProt/bin/python ./train.py \
name=${name} experiment=${exp} datamodule=${dataset}  \
logger=tensorboard trainer=ddp_fp16
