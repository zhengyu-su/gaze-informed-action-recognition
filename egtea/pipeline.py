import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.data import *
from data.load_data import *
from tqdm import tqdm
import logging
from datetime import datetime

from ultralytics import YOLO

import torch
from torch.utils.data import DataLoader

FRAME_HEIGHT = 480
FRAME_WIDTH = 640
FPS = 24

def create_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    log_file_name = f"training_log_{timestamp}.txt"
    log_file = os.path.join(log_dir, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Logger initialized. Training starts!")
    return logger



def main(args):
    # Settings DO NOT CHANGE HERE
    root_path = args.root_path
    model = YOLO("yolov8s-world.pt")

    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current Device:", torch.cuda.current_device())
        print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    model = model.to(DEVICE)
    print(f"Model loaded on {DEVICE}")

    action_idx_path = os.path.join(root_path, 'action_annotation/action_idx.txt')
    noun_idx_path = os.path.join(root_path, 'action_annotation/noun_idx.txt')
    clips_root_path = os.path.join(root_path, 'cropped_clips')
    action_to_idx, idx_to_action, action_classes, noun_classes = create_action_dict(action_idx_path, noun_idx_path)

    # Set action class for the model
    model.set_classes(noun_classes)

    # Get selected videos and experiment names
    selected_videos_train, experiment_names_train = get_video_names_split(args.root_path, 'train_split1.txt')
    selected_videos_test, experiment_names_test = get_video_names_split(args.root_path, 'test_split1.txt')

    transform = None
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((640, 640)),  # Resize to multiple of 32 for the stride in YOLO model
    #     transforms.ToTensor(), 
    # ])

    train_dataset = EGTEADataset(root_path, experiment_names_train, selected_videos_train, transform=transform)
    test_dataset = EGTEADataset(root_path, experiment_names_test, selected_videos_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # experiment_base_dir = '/Users/xiaoxiao/Documents/Studium/Lab/GazeEstimation/Results/EGTEAGaze+'
    experiment_base_dir = r"C:\Users\Su\Documents\Codes\Results\EGTEA"
    experiment_folder = create_experiment_folder(experiment_base_dir)

    logger = create_logger(log_dir=os.path.join(experiment_folder, "logs"))

    # Write the class names to current experiment folder
    class_names_path = os.path.join(experiment_folder, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for name in model.names:
            f.write(f"{model.names[name]}\n")
    logger.info(f"Class names saved to {class_names_path}")

    logger.info("Starting pipeline...")
    logger.info(f"{len(train_dataset)} samples in the training dataset")
    logger.info(f"{len(test_dataset)} samples in the test dataset")

    # Process the training set
    logger.info("Processing training set...")
    for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing training videos"):
        batch_size = len(sample['name'])

        for video_idx in range(batch_size):  # Iterate over each video in the batch
            video_name = sample['name'][video_idx]
            action_idx = sample['action_idx'][video_idx]
            gaze_data = sample['gaze'][video_idx]
            frames = sample['frames'][video_idx]  # shape: (150, H, W, C)
            action_name = idx_to_action[int(action_idx)]
            fixation_count = sample['fixation_count'][video_idx]
            
            try:
                results_video_train, results_path = process_video(
                    [{'name': video_name, 'action_idx': action_idx, 'gaze': gaze_data, 'frames': frames, 'fixation_count': fixation_count}],
                    model,
                    experiment_folder,
                    idx_to_action,
                    roi_size=args.roi,
                    save_video=args.save_video,
                    data_split='train',
                )
            except Exception as e:
                logger.error(f"Error processing video {video_name}: {e}")

    # Process the test set
    logger.info("Processing test set...")
    for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing test videos"):
        batch_size = len(sample['name'])

        for video_idx in range(batch_size):
            video_name = sample['name'][video_idx]
            action_idx = sample['action_idx'][video_idx]
            gaze_data = sample['gaze'][video_idx]
            frames = sample['frames'][video_idx]
            action_name = idx_to_action[int(action_idx)]
            fixation_count = sample['fixation_count'][video_idx]

            try:
                results_video_test, results_path = process_video(
                    [{'name': video_name, 'action_idx': action_idx, 'gaze': gaze_data, 'frames': frames, 'fixation_count': fixation_count}],
                    model,
                    experiment_folder,
                    idx_to_action,
                    roi_size=args.roi,
                    save_video=args.save_video,
                    data_split='test',
                )
            except Exception as e:
                logger.error(f"Error processing video {video_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--root_path', type=str, default='/Users/xiaoxiao/Documents/Studium/Lab/GazeEstimation/Data/EGTEAGaze+')
    # parser.add_argument('--root_path', type=str, default='/Users/xiaoxiao/Documents/Studium/Lab/GazeEstimation/Data/dataset/EGTEA')    # for local computer
    parser.add_argument('--root_path', type=str, default=r"Z:\05_Datasets\EGTEA Gaze +")    # for remote computer
    # parser.add_argument('--root_path', type=str, default='/Volumes/Public/05_Datasets/EGTEA Gaze +')  # for my own laptop
    parser.add_argument('--save_video', action='store_true', help='Save the video with bounding boxes for visualization')
    parser.add_argument('--save_labels', action='store_true', help='Save the labels to a file')
    parser.add_argument('--roi', type=tuple, default=(80,80), help='Size of the region of interest')
    # parser.add_argument('--selected_actions', type=list, default=[93, 69, 84, 86], help='Selected action indices')

    args = parser.parse_args()
    main(args)