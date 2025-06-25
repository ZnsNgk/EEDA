export OMP_NUM_THREADS=1

N_GPUS=$1
MODEL=$2    # transnext_base, transnext_eeda_base

python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
    --use_env main.py \
    --data-path /root/shared-nvme/ImageNet-1K/CLS-LOC \
    --mini-batch 32 \
    --model ${MODEL} \
    --drop-path 0.1 \
    --output_dir ./${MODEL} \
    --num_workers 18 \
    --batch-size 64\
