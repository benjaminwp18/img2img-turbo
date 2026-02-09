accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/pix2pix_turbo/phaze" \
    --dataset_folder="data/phaze" \
    --resolution=512 \
    --num_training_epochs=10000 \
    --train_batch_size=2 \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --report_to "wandb" --tracker_project_name "pix2pix_turbo_phaze"

#    --pretrained_model_name_or_path="/home/bwp18/img2img-turbo/output/pix2pix_turbo/phaze/checkpoints/model_2501.pkl" \

