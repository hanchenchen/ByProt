cp /hanchenchen/torch  /root/.cache/ -r  
env_name=ByProt 
conda activate ${env_name} 
cd  /hanchenchen/ByProt 
exp=fixedbb/lm_design_esm1b_650m 
dataset=cath_4.2 
name=fixedbb/${dataset}/lm_design_esm1b_650m  
/root/miniconda3/envs/ByProt/bin/python ./train.py \
experiment=${exp} datamodule=${dataset} name=${name} \
logger=tensorboard trainer=ddp_fp16  