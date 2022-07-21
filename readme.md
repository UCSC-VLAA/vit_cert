# Certified Detection and Recovery for Patch Attacks with Vision Transformers
This is the official implementation of the paper 'ViP: Unified Certified Detection and Recovery for Patch Attack with Vision Transformers'.

## Certified detection
In this task we use off-the-shelf models and no additional training is needed. After specifying model path and data path, run:

```
python3 main.py \
    --batch_size 1024 \
    --model vit_base_patch16 \
    --finetune '' \
    --resume '' \
    --cert /off/the/shelf/model/path \
    --output_dir ./output_dir/detection \
    --dist_eval \
    --data_path /data/path \
    --name exp_name
```

## Certified recovery
We need additional finetuning since the size of band is small. Run the following command to finetune 30 epochs
 and compute recovery-based certified accuracy. 
 ```
python3 main.py \
    --batch_size 256 \
    --epochs 30 \
    --accum_iter 1 \
    --model vit_base_patch16 \
    --weight_decay 1e-4 \
    --layer_decay 1 \
    --blr 1e-3 \
    --finetune /model/path \
    --resume '' \
    --width 19 \
    --output_dir ./output_dir/recovery/w19 \
    --name recovery_width19 \
    --data_path /data/path
```

## Acknowledgement
This repo is built on [MAE](https://github.com/facebookresearch/mae) and [smoothed-vit](https://github.com/MadryLab/smoothed-vit). And this work is supported by a gift from Open Philanthropy, TPU Research Cloud (TRC) program, and Google Cloud Research Credits program.


## Citation

```
@inproceedings{li2022vip,
  title     = {ViP: Unified Certified Detection and Recovery for Patch Attack with Vision Transformers}, 
  author    = {Junbo Li and Huan Zhang and Cihang Xie},
  booktitle = {ECCV},
  year      = {2022},
}
```
>>>>>>> 40fcb73481ebd483714ef5be8e75b21322b18e8b
