#python src/inference_unpaired.py --model_path "output/cyclegan_turbo/uhaze/checkpoints/model_25001.pkl" \
#    --input_image "data/uhaze/test_A/20251112_112422.mp4_00007.png" \
#    --prompt "object in water" --direction "a2b" \
#    --output_dir "predictions" --image_prep "resize_256"

#python src/inference_unpaired.py --model_path "output/cyclegan_turbo/uhaze/checkpoints/model_25001.pkl" \
#    --input_image "data/uhaze/test_B/w0_00005.png" \
#    --prompt "object in sand" --direction "b2a" \
#    --output_dir "predictions" --image_prep "resize_256"

python src/inference_unpaired.py --model_path "output/cyclegan_turbo/uhaze/checkpoints/model_25001.pkl" \
    --input_video "data/uhaze_for_pred/a/videos/blue_sand.mp4" \
    --prompt "object in water" --direction "a2b" \
    --output_dir "predictions" --image_prep "resize_256"

#python src/inference_unpaired.py --model_path "output/cyclegan_turbo/uhaze/checkpoints/model_25001.pkl" \
#    --input_video "data/uhaze_for_pred/b/videos/LT_35_100.mp4" \
#    --prompt "object in sand" --direction "b2a" \
#    --output_dir "predictions" --image_prep "resize_256"

