import os
import cv2
import torch
import numpy as np
from dataset import ValDataset

def apply_noise(img_train, noise_ival):
    # std dev of each sequence
    stdn = torch.empty((1, 1, 1)).cuda().uniform_(noise_ival[0], to=noise_ival[1])

    # draw noise samples from std dev tensor
    noise = torch.zeros_like(img_train)
    noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

    # define noisy input
    imgn_train = img_train + noise

    return imgn_train

def process_video(input_path, output_path, noise_ival):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to tensor
        img_train = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).float().cuda()

        # Apply noise to the frame
        imgn_train = apply_noise(img_train, noise_ival)

        # Convert back to numpy array and write to output video
        noisy_frame = imgn_train.squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        out.write(noisy_frame)

    cap.release()
    out.release()

def process_video_images_folder(input_path, output_path, noise_ival):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for filename in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path, filename))

        # Convert frame to tensor
        img_train = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().cuda()

        imgn = apply_noise(img, noise_ival)



def process_folder(input_folder, output_folder, noise_ival):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_video(input_path, output_path, noise_ival)

if __name__ == "__main__":
    input_folder = "/home/pau/TFG/tests/src_frms"
    output_folder = "/home/pau/TFG/tests/dst_frms_py"
    noise_ival = (30, 30)  # Adjust as needed

    process_folder(input_folder, output_folder, noise_ival)
