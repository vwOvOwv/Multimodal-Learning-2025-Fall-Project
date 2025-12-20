export HF_HOME="/home/data10T/lpy/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

CUDA_VISIBLE_DEVICES=3,5 accelerate launch --num_processes 2 train_lddinov2.py \
 --tracker_project_name "lddinov2" \
 --output_dir /home/data10T/lpy/mml-proj/ckpts/lddinov2 \
 --train_data_dir /home/data10T/lpy/instructclip_datasets/instructpix2pix-clip-filtered \
 --train_batch_size 16 \
 --gradient_accumulation_steps 1 \
 --dataloader_num_workers 8 \
 --max_train_steps 100000 \
 --validation_steps 10000 \
 --checkpointing_steps 10000 \
 --learning_rate 1e-5 \
 --report_to wandb
