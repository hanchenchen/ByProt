pip install -e .
dataset=af2
# name=fixedbb/${dataset}/protein_mpnn_cmlm
name=fixedbb/${dataset}/0913_lm_design_esm2_650m
exp_path=/hanchenchen/byprot1/ByProt/logs/${name}

python ./test.py \
experiment_path=${exp_path} \
data_split=test ckpt_path=best.ckpt mode=predict \
task.generator.max_iter=5