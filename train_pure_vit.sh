export OMP_NUM_THREADS=1

N_GPUS=$1
MODEL_CFG=$2    # model_config file name, do not add ".json"

python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
    --use_env main.py \
    --data-path /root/shared-nvme/ImageNet-1K/CLS-LOC \
    --mini-batch 64 \
    --model pure_vit \
    --model-cfg ./model_config/${MODEL_CFG}.json \
    --drop-path 0.1 \
    --num_workers 18 \
    --batch-size 128 \
    --output_dir ./${MODEL_CFG}