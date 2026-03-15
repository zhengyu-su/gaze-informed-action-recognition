import pandas as pd
import os
from datasets import Dataset
import numpy as np
import torch
import cv2
from gaze_io_sample import parse_gtea_gaze
from PIL import ImageColor
import csv
from torchvision import transforms
from tqdm import tqdm

FRAME_HEIGHT = 480
FRAME_WIDTH = 640
FPS = 24
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

COLORS={
'ZEISSIndigo': '#0f2db3',
'ZEISSAqua': '#244a86',
'ZEISSSaphire': '#4c6bb1',
'ZEISSAzur': '#0072ef',
'ZEISSCyan': '#008bd0',
'ZEISSSkyBlue': '#6ab0e2',
'ZEISSSteel': '#8daac8',
'ZEISSArctic': '#c6daf2',
'ZEISSGray0Ultradark': '#32373e',
'ZEISSGray1Dark': '#4e565f',
'ZEISSGray2Semidark': '#606a76',
'ZEISSGray3Medium': '#778592',
'ZEISSGray4Semilight': '#929eab',
'ZEISSGray5Light': '#b4c0ca',
'ZEISSGray6Ultralight': '#dce3e9',
'ZEISSGray7Semiwhite': '#f2f5f8',
'ZEISSBrightOrange': '#e71e1e',
'ZEISSPurpleRed': '#a70240',
'ZEISSGreen': '#1e8565',
'ZEISSLightGreen': '#d9e906',
'ZEISSBrightLemon': '#fdbb08',
'ZEISSOrange': '#ea591b'}

def get_video_names():
    '''
    Get the video names+index paris and the experiment names for selected actions from the train test split
    Returns:
        - selected_videos:
        A list of dictionaries with keys: name, action_idx
        - experiment_names:
        A list of experiment names (without the frame indices)
    '''
    # selected action names: ["crack egs", "put cheese", "take cheese", "mix egg"]
    split_root = '/Volumes/Public/05_Datasets/EGTEA Gaze +/action_annotation'
    split_paths = [os.path.join(split_root, f) for f in os.listdir(split_root) if f.startswith('train')]

    ### Remove magic numbers!
    selected_action_idx = [93, 69, 84, 86]     # Action indices that we want to analyze
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
            if experiment_name not in experiment_names:
                experiment_names.append(experiment_name)
            first_number = int(parts[1])      # action index

            # Check if the current action index is in the selected action index
            # And not alrready in the selected videos
            if int(first_number) in selected_action_idx:
                if not any(video['name'] == name_part for video in selected_videos):
                    video = {'name': name_part, 'action_idx': first_number}
                    selected_videos.append(video)
    return selected_videos, experiment_names

def load_token_data(root_path):
    '''
    Load data to pandas dataframe
    '''
    data = []
    id2label = {}
    label2id = {}

    # Get the action classes
    action_classes = [f for f in os.listdir(root_path) if not f.endswith('.txt') and not f.startswith('.') and not f.startswith('Plot')]

    for i, action_class in enumerate(action_classes):
        id2label[i] = action_class
        label2id[action_class] = i
        action_class_path = os.path.join(root_path, action_class)
        if os.path.isdir(action_class_path):
            for video_clip in os.listdir(action_class_path):
                video_clip_path = os.path.join(action_class_path, video_clip)
                if os.path.isdir(video_clip_path):
                    # Look for the token_scan_path.csv file
                    for file in os.listdir(video_clip_path):
                        if file.endswith("_token_scan_path.csv"):
                            csv_path = os.path.join(video_clip_path, file)
                                
                            # Only need the column for the token scan path
                            token_scan_df = pd.read_csv(csv_path)['Label'].tolist()
                                
                            # Append a record with action_class, video_clip, and token_scan_path content
                            data.append({
                                "action_class": action_class,
                                "action_idx": i,
                                "video_clip": video_clip,
                                "token_scan_path": token_scan_df  # Storing the whole DataFrame
                            })

    df = pd.DataFrame(data)
    return df, id2label, label2id, action_classes

def sort_fixations(gaze_data):
    '''
    Returns:
    sorted_fixations: dictionary of sorted fixations
    every continuous fixation is grouped together and the index in the whole video
    is saved
    '''
    gaze_types = gaze_data[:, 2]   # Check for the gaze type column
    fixations = {}
    fixation_count = 0
    current_indices = []

    for idx, value in enumerate(gaze_types):
        if value == 1:
            current_indices.append(idx)
        else:
            if current_indices:
                fixation_count += 1
                fixations[f'fixation_{fixation_count}'] = current_indices
                current_indices = []

    if current_indices:
        fixation_count += 1
        fixations[f'fixation_{fixation_count}'] = current_indices

    return fixations

def get_scan_path(fixations_dict, token_scan_path, fixation_count):
    '''
    Use winner take all to get the fixations scan path

    Input:
    fixations_dict:
    {
        fixation_1: {
            "label": [class1, class2, class3],
            "duration": 2.5,
            "gaze_locations": [(x1, y1), (x2, y2), (x3, y3)]
        },
        fixation_2: {
            "label": [class1, class2, class3],
            "duration": 1.5,
            "gaze_locations": [(x1, y1), (x2, y2), (x3, y3)]
        },
        ...
    }
    '''
    scan_path_file = os.path.join(token_scan_path)
    placeholder = 'x'
    scan_path = {}

    # Open the CSV file in write mode
    with open(scan_path_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Fixation", "Label", "Duration", "Gaze Location", "Fixation Count"])  # Write header

        # for idx, value in fixations_dict.items():
        #     if value:
        #         winner = max(set(value), key=value.count)  # Get the most frequent class
        #         scan_path[idx] = winner
        #         writer.writerow([idx, winner,])  # Write fixation and winner to CSV
        #     else:
        #         scan_path[idx] = placeholder
        #         writer.writerow([idx, placeholder])  # Write placeholder if no winner
        for fixation_idx, values in fixations_dict.items():
            labels = values["label"]
            duration = values["duration"]
            gaze_locations = values["gaze_locations"]
            gaze_locations_str = ', '.join([f"({gaze_x}, {gaze_y})" for gaze_x, gaze_y in gaze_locations])

            if labels:
                winner = max(set(labels), key=labels.count)
            else:
                winner = placeholder
            scan_path[fixation_idx] = winner
            writer.writerow([fixation_idx, winner, duration, gaze_locations_str, fixation_count])  # Write fixation, winner, duration, and gaze locations to CSV

            # breakpoint()
                    
    return scan_path

def get_data(root_path, experiment_names, selected_videos,):
    '''
    Returns a list of dictionaries of gaze data for selected videos
        selected_data = [{
            'name': video_name,  whole name including the frame indices (the .mp4 names)
            'action_idx': action_idx,   can be transformed to action name
            'gaze': selected_gaze_data,   [norm_x, norm_y, gaze_type, 0]
            'frames': video_frames
        }]
    '''
    gaze_root = os.path.join(root_path, 'gaze_data/gaze_data/')
    video_root = os.path.join(root_path, 'video_clips/cropped_clips')
    experiments = [f for f in experiment_names]
    # get selected gaze data
    gaze_data = {}
    for exp_name in experiments:
        gaze_data[exp_name] = parse_gtea_gaze(os.path.join(gaze_root, exp_name+'.txt'))
        print('Loaded gaze data from {:s}'.format(exp_name))
        # Comment out at the end
        # if len(gaze_data) == 10:
        #     break
    
    # Select the fixation gaze data
    selected_data = []
    for video in selected_videos:
        video_name_split = video['name'].split('-')
        video_name = '-'.join(video_name_split[:3])
        frame_start = int(video_name_split[-2].split('F')[-1])
        frame_end = int(video_name_split[-1].split('F')[-1])
        # print(f'start frame: {frame_start}, end frame: {frame_end}, total frames: {frame_end-frame_start+1}')
        action_idx = video['action_idx']

        if video_name in gaze_data.keys():
            selected_gaze_data = gaze_data[video_name][frame_start:frame_end+1]
            # print(len(selected_gaze_data))

            # Check if there are any fixation data, if not, skip the video
            has_fixation = np.any(selected_gaze_data[:, 2] == 1)

            skipped_videos = []

            if not has_fixation:
                print(f'Video {video_name} has no fixation data, skipping...')
                skipped_videos.append(video_name)

            else:
                # Get the video frames
                video_path = os.path.join(video_root, video_name, video['name']) + '.mp4'
                cap = cv2.VideoCapture(video_path)
                video_frames = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    video_frames.append(frame)
                    # print(f'frame type: {type(frame)}')

                cap.release()
                print(f"Video {video['name']} loaded")

                # Make sure that the length of the gaze data is the same as the video frames
                # assert len(selected_gaze_data) == len(video_frames), f'Length of gaze data: {len(selected_gaze_data)} and video frames: {len(video_frames)} from {video['name']} are not the same!'

                if len(selected_gaze_data) == len(video_frames):
                    selected_data.append({'name': video['name'], 'action_idx': action_idx, 
                                          'gaze': selected_gaze_data,
                                          'frames': video_frames,})
                
                # print(f'length of gaze data: {len(selected_gaze_data)}')
                # print(f'length of video frames: {len(video_frames)}')
                # print(f'example of gaze data: {selected_gaze_data[0]}')
                # print(f'example of a frame data: {video_frames[0]}')
            # print(f'selected_data: {selected_data}')
            print(f"{len(skipped_videos)} videos skipped: {skipped_videos}")
    return selected_data

def process_video(selected_data, model, experiment_folder, index_to_action, roi_size=(100,100), save_video=False, save_labels=True, data_split='train'):
    '''
    Process the selected videos: first detect the objects in the whole video,
    then extract the ROI based on the gaze information

    Returns:
    fixation data for the DistilBERT model that is a fixation
    fixation_data = {
        'video_name': video_name,
        'action_idx': action_idx,
        'token_ids': token_ids,
        'labels': labels
    }
    '''

    # print(f'Processing {len(selected_data)} selected videos...')

    color_box = ImageColor.getcolor(COLORS['ZEISSGray7Semiwhite'], 'RGB')[::-1] # Convert to BGR for OpenCV
    color_gaze = ImageColor.getcolor(COLORS['ZEISSBrightOrange'], 'RGB')[::-1]

    # color = (255, 255, 255)

    for i, data in enumerate(selected_data):
        # print(f'Processing video {i+1}/{len(selected_data)}')

        # breakpoint()

        video_name = data['name']
        action_idx = int(data['action_idx'])
        action_name = index_to_action[action_idx]
        gaze_data = data['gaze']
        video_frames = data['frames']
        sorted_fixations = sort_fixations(gaze_data)
        fixation_count = data['fixation_count']

        # Path to save the results for each video
        # If there is a '/' in the action name, replace it with '_'
        action_name = action_name.replace('/', '_')
        results_path = os.path.join(experiment_folder, data_split, action_name, video_name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        labels_path = os.path.join(results_path, f'{action_name}_labels.txt')     # file to save the token scanpath
        token_scan_path = os.path.join(results_path, f'{action_name}_token_scan_path.csv')     # file to save the token scanpath

        # Initialize the video writer
        # height, width, _ = video_frames[0].shape
        # print(f'height: {height}, width: {width}')  height=480, width=640

        results_video = {}

        # fixations_to_labels = {fixation: [] for fixation in sorted_fixations.keys()}
        fixations_dict = {fixation: 
                               {"label": [],
                                "duration": len(sorted_fixations[fixation])/FPS,
                                "gaze_locations": []}
                                for fixation in sorted_fixations}

        if save_video == True:
            video_path = os.path.join(results_path, video_name + '_results.mp4')
            fps = 10
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

        # Process the video frames
        for idx, frame in enumerate(video_frames):
            # Get gaze type and location
            gaze_type = gaze_data[idx][2]  # 1 for fixation
            gaze_x, gaze_y = gaze_data[idx][:2]

            frame_copy = frame.clone()
            frame_copy = frame_copy.cpu().numpy()
            frame_copy = frame_copy.astype(np.uint8)    # uint8 for OpenCV
            # frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)

            # Rescale the gaze location and add the gaze position to the frame 
            gaze_x, gaze_y = gaze_x*FRAME_WIDTH, gaze_y*FRAME_HEIGHT            
            
            cv2.circle(frame_copy, (int(gaze_x), int(gaze_y)), radius=10, color=color_gaze, thickness=-1)

            # Get current fixation
            current_fixation = None
            
            for fixation, indices in sorted_fixations.items():
                if idx in indices:
                    current_fixation = fixation
                    break
            
            # print(f'Frame {idx}: Fixation {current_fixation}')

            # Only process the frame for object detection if it is a fixation
            if gaze_type == 1 and current_fixation:

                # print(f'Processing fixation {current_fixation} at frame {idx}')

                fixations_dict[current_fixation]["gaze_locations"].append((gaze_x, gaze_y))

                # Get the object detection results
                frame = frame.float().to(DEVICE)/255.0
                frame = frame.permute(2, 0, 1).unsqueeze(0)
                
                result = model(frame, conf=0.05, iou=0.15, verbose=False)

                # frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

                # Add the bounding boxes to the frame
                for box in result[0].boxes:
                    # calculate ROI based on the gaze position
                    x_roi = max(0, int(gaze_x - roi_size[0] // 2))
                    y_roi = max(0, int(gaze_y - roi_size[1] // 2))
                    w_roi, h_roi = roi_size
                    # bounding box (x_center, y_center, width, height)
                    x, y, w, h = box.xywh[0].cpu().numpy()

                    cls_idx = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls_idx]
                    confidence = box.conf[0].cpu().numpy()

                    # Check if the center of the bounding box is within the ROI
                    if  x_roi < x < x_roi + w_roi and y_roi < y < y_roi + h_roi:

                        # print(f'Object detected: {class_name} with confidence: {confidence:.2f}')

                        fixations_dict[current_fixation]["label"].append(class_name)

                        # Write the label to the file
                        if save_labels == True:

                            with open(labels_path, 'a') as f:
                                f.write(f"{class_name}\n")

                        # Calculate the two corners of the rectangle
                        pt_1 = (int(x - w//2), int(y - h//2))
                        pt_2 = (int(x + w//2), int(y + h//2))

                        # Draw the rectangle on the original frame
                        cv2.rectangle(frame_copy, pt_1, pt_2, color_box, 5)

                        # Combine class name and confidence into a label string
                        label = f"{class_name} ({confidence:.2f})"

                        # Add class name text near the bounding box
                        text_position = (pt_1[0], pt_1[1] - 10)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        thickness = 2
                            
                        # Add the text to the frame
                        cv2.putText(frame_copy, label, text_position, font, font_scale, color_box, thickness)

            # Write after all the bounding boxes are drawn
            if save_video == True:
                out.write(frame_copy)
        # print(f'fixations_to_labels: {fixations_to_labels}')

        # breakpoint()
        _ = get_scan_path(fixations_dict, token_scan_path, fixation_count)   # Write the scan path to a CSV file

    # Release the video writer at the end of the event
    if save_video == True:
        out.release()

    return results_video, results_path