## E^2DA: An Efficient Plug-and-play Feed-Forward Network for Lightweighting Visual Transformers

This is the official repository for Efficient Embedding Dimensional Attention (E^2DA) Module.

### Requirements:

* torch>=2.0+
* torchvision>=0.15.2+
* timm>=0.3.2
* torch_flops

### Install
Before running TransNext for training, you should first install the `swattention_extension` package. After installing pytorch, nvcc, and some necessary environments, perform the following steps:


````bash
cd ./swattention_extension
python setup.py install
````

### Train

Use the following commands to train `Vit w/. E^2DA` for distributed learning with 4 GPUs:

````bash
python -m torch.distributed.launch --nproc_per_node=4 \
    --use_env main.py \
    --data-path ${your dataset path} \
    --mini-batch 64 \
    --model pure_vit \
    --model-cfg ./model_config/pure_vit_eeda.json \
    --drop-path 0.1 \
    --num_workers 18 \
    --batch-size 128 \
    --output_dir ./pure_vit_eeda
````


In addition, we provide one click code execution, and you should modify the dataset path information in the `.sh` file before running, as follows:


````bash
sh ./train_pure_vit.sh 4 pure_vit_eeda
````

In addition, the number of module replacements and all generalization experiments have been provided, and you can train by running the relevant sh files. It should be noted that these experiments will consume a significant amount of time.

Number of module replacements:

````bash
sh ./run_pure_vit_forward.sh
sh ./run_pure_vit_backward.sh
````

Generalization experiments (change the dataset path in the sh file before running):

````bash
# poolformer
sh  ./train_poolformer.sh 4 poolformer_m48
sh  ./train_poolformer.sh 4 poolformer_eeda_m48

# cpvt
sh ./train_cpvt.sh 4 pcpvt_base_v0
sh ./train_cpvt.sh 4 pcpvt_eeda_base_v0

# nextvit
sh ./train_nextvit.sh 4 nextvit_base
sh ./train_nextvit.sh 4 nextvit_eeda_base

# p2t
sh ./train_p2t.sh 4 p2t_base
sh ./train_p2t.sh 4 p2t_eeda_base

# transnext
sh ./train_transnext.sh 4 transnext_base
sh ./train_transnext.sh 4 transnext_eeda_base
````