from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from gaze_io_sample import *
import torch

# Dataloader and other helper functions

class EGTEADataset(Dataset):
    def __init__(self, root_path, experiment_names, selected_videos, max_video_len=150, transform=None):
        """
        Args:
            root_path (str): root path of the dataset
            experiment_names (list): 
            selected_videos (list): selected videos
            transform (callable, optional): data augmentation
        """
        self.root_path = root_path
        self.experiment_names = experiment_names
        self.selected_videos = selected_videos
        self.transform = transform
        # self.data = self._load_data()

        self.gaze_data = self._load_gaze_data()
        self.video_paths = self._get_video_paths()
        self.max_video_len = max_video_len

    def _load_gaze_data(self):
        """
        Load gaze data from the dataset.
        """
        gaze_root = os.path.join(self.root_path, 'gaze_data/gaze_data/')
        gaze_data = {}
        print("Loading gaze data...")
        for exp_name in self.experiment_names:
            gaze_data[exp_name] = parse_gtea_gaze(os.path.join(gaze_root, exp_name + '.txt'))
        print("Finished loading gaze data!")
        return gaze_data
    
    def _get_video_paths(self):
        """
        Get the paths of the videos.
        """
        video_root = os.path.join(self.root_path, 'cropped_clips')
        if not os.path.exists(video_root):
            video_root = os.path.join(self.root_path, 'video_clips/cropped_clips')
        video_paths = []
        for video in self.selected_videos:
            video_name_split = video['name'].split('-')
            video_name = '-'.join(video_name_split[:3])
            frame_start = int(video_name_split[-2].split('F')[-1])
            frame_end = int(video_name_split[-1].split('F')[-1])
            action_idx = video['action_idx']

            # if video_name not in self.gaze_data.keys():
            #     print(f"Video {video_name} not found in gaze data.")

            if video_name in self.gaze_data.keys():
                selected_gaze_data = self.gaze_data[video_name][frame_start:frame_end + 1]
                has_fixation = np.any(selected_gaze_data[:, 2] == 1)

                # if not has_fixation:
                #     print(f"No fixation found for video {video['name']}")
                if has_fixation:
                    video_path = os.path.join(video_root, video_name, video['name']) + '.mp4'
                    if os.path.exists(video_path):
                        video_paths.append({
                            'name': video['name'],
                            'action_idx': action_idx,
                            'gaze': selected_gaze_data,
                            'path': video_path,
                        })
                else:
                    # print(f"Video {video['name']} has no fixation")
                    continue
                
            else:
                print(f"Video {video_name} not found in gaze data.")
        
        return video_paths

    def _load_data(self):
        """
        Load data and returns a list, each element is a dictionary containing 
        video frames and gaze data.
        """
        gaze_root = os.path.join(self.root_path, 'gaze_data/gaze_data/')
        video_root = os.path.join(self.root_path, 'cropped_clips')
        gaze_data = {}
        print("Loading gaze data...")
        for exp_name in self.experiment_names:
            gaze_data[exp_name] = parse_gtea_gaze(os.path.join(gaze_root, exp_name + '.txt'))
        print("Finished loading gaze data!")

        selected_data = []
        for video in self.selected_videos:
            video_name_split = video['name'].split('-')
            video_name = '-'.join(video_name_split[:3])
            frame_start = int(video_name_split[-2].split('F')[-1])
            frame_end = int(video_name_split[-1].split('F')[-1])
            action_idx = video['action_idx']

            if video_name in gaze_data.keys():
                selected_gaze_data = gaze_data[video_name][frame_start:frame_end + 1]
                has_fixation = np.any(selected_gaze_data[:, 2] == 1)

                if has_fixation:
                    video_path = os.path.join(video_root, video_name, video['name']) + '.mp4'
                    if os.path.exists(video_path):
                        cap = cv2.VideoCapture(video_path)
                        video_frames = []
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            video_frames.append(frame)
                        cap.release()

                        if len(selected_gaze_data) == len(video_frames):
                            selected_data.append({
                                'name': video['name'],
                                'action_idx': action_idx,
                                'gaze': selected_gaze_data,
                                'frames': video_frames,
                                'length': len(video_frames),
                            })
                        else:
                            print(f"Length of gaze data ({len(selected_gaze_data)}) and video frames ({len(video_frames)}) do not match for video {video['name']}")
        return selected_data

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_info = self.video_paths[idx]
        video_path = video_info['path']
        gaze_data = video_info['gaze']
        seq_len = len(gaze_data)  # Original video length
        fixation_count = np.sum(gaze_data[:, 2] == 1)


        # Load data to max_video_len or if shorter then to its original length
        end_len = min(seq_len, self.max_video_len)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True and len(frames) < end_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = torch.tensor(frame, dtype=torch.float32)
            frames.append(frame)
        cap.release()

        if seq_len > self.max_video_len:
        # Cut the video to the max_video_len
            gaze_data = gaze_data[:self.max_video_len]
        elif seq_len < self.max_video_len:
        # Pad the video to the max_video_len
            gaze_data = np.pad(gaze_data, ((0, self.max_video_len - seq_len), (0, 0)), 'constant', constant_values=0)
            last_frame = frames[-1] if len(frames) > 0 else torch.zeros_like(frames[0])
            frames += [last_frame] * (self.max_video_len - seq_len)
            # frames = frames + [frames[-1]] * (self.max_video_len - seq_len)
            
        frames = torch.stack(frames)

        return {
            'name': video_info['name'],
            'action_idx': video_info['action_idx'],
            'gaze': gaze_data,
            'frames': frames,
            'length': seq_len,
            'fixation_count': fixation_count,
        }

def get_video_names_split(root_path, split_name):
    '''
    Get all the video names+index paris and the experiment names for actions from the train split 1
    Returns:
        - selected_videos:
        A list of dictionaries with keys: name, action_idx
        - experiment_names:
        A list of experiment names (without the frame indices)
    '''
    # split_root = '/Volumes/Public/05_Datasets/EGTEA Gaze +/action_annotation'
    split_root = os.path.join(root_path, 'action_annotation')
    # split_paths = [os.path.join(split_root, f) for f in os.listdir(split_root) if f.startswith('train')]
    split_paths = [os.path.join(split_root, split_name)]

    print(f'split_paths: {split_paths}')

    # Check if the split file exists
    for split_path in split_paths:
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file {split_path} not found.")

    selected_videos = []
    experiment_names = []
    for split_path in split_paths:
        with open(split_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            parts = line.split()
            # P21-R06-GreekSalad-899463-903560-F021577-F021696 for the mp4 
            # video clip file
            name_part = parts[0]         # name for the video clip
            experiment_name = '-'.join(name_part.split('-')[:3])
            # Add the used split to the experiment names
            # experiment_name += '_train_split1'
            if experiment_name not in experiment_names:
                experiment_names.append(experiment_name)
            first_number = int(parts[1])      # action index

            if not any(video['name'] == name_part for video in selected_videos):
                video = {'name': name_part, 'action_idx': first_number}
                selected_videos.append(video)
    # print(f'experiment_names: {experiment_names}')
    return selected_videos, experiment_names

def create_experiment_folder(base_dir):
    """
    Creates an experiment folder with an incrementing number (e.g., experiment_0, experiment_1).
    """
    experiment_counter = 0
    while True:
        experiment_folder = os.path.join(base_dir, f'experiment_{experiment_counter}')
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
            break
        experiment_counter += 1
    return experiment_folder

# Function to read the file and convert to dictionary
def create_action_dict(action_path, noun_path):
    '''
    Function to give the action and noun classes.

    Parameters:
        action_path: path to the action classes file
        noun_path: path to the noun classes file
    Returns:
        action_to_idx: dictionary of action classes to index
        idx_to_action: dictionary of index to action classes
        action_classes: list of action classes
        noun_classes: list of noun classes (used later for setting the model classes)
    '''
    action_to_idx = {}
    idx_to_action = {}
    action_classes = []
    noun_classes = []
    with open(action_path, 'r') as file:
        for line in file:
            action, index = line.rsplit(maxsplit=1)
            action_to_idx[action] = int(index)
            idx_to_action[int(index)] = action
            action_classes.append(action)
    
    # noun classes
    with open(noun_path, 'r') as file:
        for line in file:
            noun, index = line.rsplit(maxsplit=1)
            noun_classes.append(noun)

    return action_to_idx, idx_to_action, action_classes, noun_classes