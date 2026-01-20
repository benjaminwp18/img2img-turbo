import os
import argparse
from PIL import Image
import cv2
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_image', type=str, help='path to the input image')
    group.add_argument('--input_video', type=str, help='path to the input video')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')

    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'

    # initialize the model
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()

    T_val = build_transform(args.image_prep)

    if args.input_image is not None:
        input_image = Image.open(args.input_image).convert('RGB')
        # translate the image
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            if args.use_fp16:
                x_t = x_t.half()
            output = model(x_t, direction=args.direction, caption=args.prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

        # save the output image
        bname = os.path.basename(args.input_image)
        os.makedirs(args.output_dir, exist_ok=True)
        output_pil.save(os.path.join(args.output_dir, bname))
    else:
        video_cap = cv2.VideoCapture(args.input_video)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        os.makedirs(args.output_dir, exist_ok=True)
        bname = Path(args.input_video).stem + '.avi'
        video_writer = cv2.VideoWriter(os.path.join(args.output_dir, bname),
                              fourcc, fps, (width, height))

        with torch.no_grad():
            current_frame = 0
            success = True

            while success:
                success, frame_cv = video_cap.read()
                if success:
                    print(f'Processing frame {current_frame}/{total_frames}')

                    frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame_cv)

                    input_img = T_val(frame)
                    x_t = transforms.ToTensor()(input_img)
                    x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
                    if args.use_fp16:
                        x_t = x_t.half()
                    output_frame = model(x_t, direction=args.direction, caption=args.prompt)

                    output_pil = transforms.ToPILImage()(output_frame[0].cpu() * 0.5 + 0.5)
                    output_pil = output_pil.resize((frame.width, frame.height), Image.LANCZOS)
                    output_cv = np.array(output_pil)
                    output_cv = output_cv[:, :, ::-1].copy()  # Convert RGB to BGR
                    video_writer.write(output_cv)

                    current_frame += 1

        video_cap.release()
        video_writer.release()
