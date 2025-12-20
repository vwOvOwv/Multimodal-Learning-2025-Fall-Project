export HF_HOME="/home/data10T/lpy/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=5

accelerate launch --num_processes 1 train_iclip.py \
 --tracker_project_name "instructclip" \
 --output_dir /home/data10T/lpy/mml-proj/ckpts/instructclip \
 --train_data_dir /home/data10T/lpy/instructclip_datasets/instructpix2pix-clip-filtered \
 --train_batch_size 16 \
 --gradient_accumulation_steps 2 \
 --dataloader_num_workers 8 \
 --max_train_steps 100000 \
 --validation_steps 10000 \
 --checkpointing_steps 10000 \
 --learning_rate 1e-5 \
 --report_to wandb \
 --resume_from_checkpoint latest
