pip install -e .
cp /hanchenchen/torch  /root/.cache/ -r  
dataset=af2_plddt70
# dataset=af2
# dataset=cath_4.2
# name=fixedbb/${dataset}/protein_mpnn_cmlm
name=fixedbb/${dataset}/0912_lm_design_esm1b_650m
exp_path=/hanchenchen/byprot1/ByProt/logs/${name}

python ./test.py \
experiment_path=${exp_path} \
data_split=test ckpt_path=best.ckpt mode=predict \
task.generator.max_iter=5