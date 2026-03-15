from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import argparse
import sys
import os
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time
import re
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.sequence_models import ActionClassifier, SimpleActionClassifier, SimpleDurationActionClassifier
from pipeline import create_logger

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(root_path):
    '''
    Load data to pandas dataframe
    '''
    id2label = {}
    label2id = {}
    # allowed_classes = ['Inspect_Read recipe', 'Open fridge', 'Take eating_utensil', 'Cut tomato', 'Turn on faucet']
    # allowed_classes = ['Inspect_Read recipe', 'Open fridge', 'Take eating_utensil']
    # allowed_classes = ['Inspect_Read recipe', 'Open fridge']
    # allowed_classes = ['Inspect_Read recipe', 'Open fridge', 'Take eating_utensil', 'Cut tomato']
    allowed_classes = None

    df_train, action_classes_train = load_data_split(root_path, "train", id2label, label2id, allowed_classes)
    df_test, action_classes_test = load_data_split(root_path, "test", id2label, label2id, allowed_classes)

    return df_train, id2label, label2id, action_classes_train, df_test, action_classes_test

def load_data_split(root_path, data_split, id2label, label2id, allowed_classes=None):
    data = []
    split_path = os.path.join(root_path, data_split)
    if allowed_classes is None:
        action_classes = sorted([
            f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))
        ])  # Get all action classes in the split directory, avoid hidden files debugging purposes, remove in production
    else:
        action_classes = sorted([
    f for f in os.listdir(split_path)
    if os.path.isdir(os.path.join(split_path, f)) and 
    (allowed_classes is None or f in allowed_classes)
    ])  # Get all action classes in the split directory, avoid hidden files debugging purposes, remove in production
    for action_class in action_classes:
        if action_class not in label2id:
            new_id = len(label2id)
            label2id[action_class] = new_id
            id2label[new_id] = action_class

        action_idx = label2id[action_class]

        action_class_path = os.path.join(split_path, action_class)
        # Check if the action class path is a directory
        try:
            action_class_path = os.path.join(split_path, action_class)
        except Exception as e:
            print(f"Error processing action class {action_class}: {e}")
            continue

        if os.path.isdir(action_class_path):
            for video_clip in os.listdir(action_class_path):
                video_clip_path = os.path.join(action_class_path, video_clip)
                if os.path.isdir(video_clip_path):
                    for file in os.listdir(video_clip_path):
                        if file.endswith("_token_scan_path.csv"):
                            csv_path = os.path.join(video_clip_path, file)
                            csv_file = pd.read_csv(csv_path)

                            # Extract the token scan paths, duration, and gaze locations
                            token_scan_df = csv_file['Label'].tolist()
                            duration_df = csv_file['Duration'].tolist()
                            fixation_count_str = csv_file['Fixation Count'].unique()[0]
                            f_count_int = int(re.search(r'\d+', fixation_count_str).group())
                            gaze_loc_df = [parse_gaze_location(gaze_location) for gaze_location in csv_file['Gaze Location'].tolist()]

                            # Create a DataFrame for the token scan paths
                            data.append({
                                "action_class": action_class,
                                "action_idx": action_idx,
                                "video_clip": video_clip,
                                "token_scan_path": token_scan_df,
                                "duration": duration_df,
                                "gaze": gaze_loc_df,
                                "fixation_count": f_count_int
                            })


    df = pd.DataFrame(data)
    return df, action_classes

def parse_gaze_location(gaze_locations_str):
    '''
    Function to parse string gaze location to a list of tuples
    '''
    gaze_locations_str = gaze_locations_str.strip("[]").strip("()")
    pairs = gaze_locations_str.split("), (")
    gaze_locations = []

    for pair in pairs:
        x, y = pair.strip("()").split(",") 
        gaze_locations.append((float(x), float(y)))
    return gaze_locations

def augment_token_sequence(tokens, p_drop=0.1, p_swap=0.05):
    # Drop some tokens
    tokens = [tok for tok in tokens if random.random() > p_drop]
    
    # Swap adjacent tokens randomly
    if len(tokens) >= 2 and random.random() < p_swap:
        i = random.randint(0, len(tokens) - 2)
        tokens[i], tokens[i+1] = tokens[i+1], tokens[i]
    
    return tokens

def get_token_ids(token_scan_paths, max_length=23):
    '''
    Args:
    token_scan_paths: List of tokenized scan paths, e.g., [["cup", "bottle"], ["plate", "fork", "cup"]]
    max_length: Maximum length to pad/truncate the token sequences

    Returns:
    padded_token_ids: List of mapped token ids (small vocab) with padding and truncation
    token2id: Mapping from original BERT token ID to new continuous ID
    '''
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Augment token sequences
    # token_scan_paths = [augment_token_sequence(seq) for seq in token_scan_paths]

    # Replace 'x' with '[PAD]' if it exists in the token scan paths
    token_scan_paths = [[tok if tok != 'x' else '[PAD]' for tok in scan_path] for scan_path in token_scan_paths]

    # Add special tokens
    special_scan_paths = [['[CLS]'] + scan_path + ['[SEP]'] for scan_path in token_scan_paths]

    # Flatten to get all unique tokens
    all_tokens = list(set([tok for scan_path in special_scan_paths for tok in scan_path]))

    # Convert to original BERT token IDs (we assume each object is one word)
    bert_token_ids = {token: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token)[0]) 
                      for token in all_tokens}

    # Sort and re-map to new continuous IDs
    sorted_bert_ids = sorted(set(bert_token_ids.values()))

    token2id = {bert_id: idx for idx, bert_id in enumerate(sorted_bert_ids)}
    id2token = {v: k for k, v in token2id.items()}

    # Special tokens handling (map them as well)
    special_tokens = {
        tokenizer.cls_token_id: token2id[101],  # [CLS]
        tokenizer.sep_token_id: token2id[102],  # [SEP]
        tokenizer.pad_token_id: token2id[0]  # [PAD]
    }
    token2id.update(special_tokens)

    # Final vocab size
    vocab_size = len(token2id)

    padded_token_ids = []
    for scan_path in special_scan_paths:
        # Tokenize → convert to original BERT IDs
        bert_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tok)[0]) for tok in scan_path]
        
        # Map to new small vocab
        mapped_ids = [token2id[bid] if bid in token2id else token2id[tokenizer.pad_token_id] for bid in bert_ids]

        # Pad/truncate
        if len(mapped_ids) < max_length:
            mapped_ids += [token2id[tokenizer.pad_token_id]] * (max_length - len(mapped_ids))
        else:
            mapped_ids = mapped_ids[:max_length]

        padded_token_ids.append(mapped_ids)

    return padded_token_ids, token2id, id2token, vocab_size


def get_token_ids_2(token_scan_paths, max_length=23):
    '''
    Args:
    token_scan_paths: List of tokenized scan paths
    max_length: Maximum length of the tokenized scan paths

    Returns:
    padded_token_ids: List of token ids with padding and truncation 
    after mapping to the corresponding token ids
    '''
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    # Replace 'x' with '[PAD]' if it exists in the token scan paths
    token_scan_paths = [[tok if tok != 'x' else '[PAD]' for tok in scan_path] for scan_path in token_scan_paths]
    # token_ids = [tokenizer(scan_path, is_split_into_words=True, max_length=150, padding="max_length", truncation=True) for scan_path in token_scan_paths]  # Assumes the scan paths are pre-tokenized
    special_scan_paths = [ ['[CLS]'] + scan_path + ['[SEP]']
                          for scan_path in token_scan_paths]  # Add special tokens
    token_ids = [tokenizer.convert_tokens_to_ids(scan_path) for scan_path in special_scan_paths]

    # Padding and truncation
    padded_token_ids = [
    ids + [tokenizer.pad_token_id] * (max_length - len(ids)) if len(ids) < max_length else ids[:max_length]
    for ids in token_ids
    ]
    

    return padded_token_ids

def get_attention_masks(token_ids):
    '''
    Args:
    token_ids: List of token ids

    Returns:
    attention_masks: List of attention masks: 1 where there is a token, 0 otherwise
    '''
    attention_masks = [[int(token_id > 0) for token_id in ids] for ids in token_ids]
    return attention_masks

def tokenize_data(padded_token_ids, attention_masks, labels):
    padded_token_ids = torch.tensor(padded_token_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    tokenized_input = {
        "input_ids": padded_token_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }
    return tokenized_input

def get_gaze_len (gaze_locations):

    all_lengths = []

    for video in gaze_locations:
        for gaze in video:
            all_lengths.append(len(gaze))
    
    percentile_90 = int(np.percentile(all_lengths, 90))
    percentile_95 = int(np.percentile(all_lengths, 95))

    return percentile_90, percentile_95

def get_padded_gaze(gaze_locations, percentile_90, max_dur_length):
    gaze_padded = []

    for video in gaze_locations:
        padded_video = []
        for gaze in video:
            padded_gaze = gaze + [(0.0, 0.0)] * (percentile_90 - len(gaze))
            padded_video.append(padded_gaze)
        # padded_video.append(padded_video)
        padded_video += [[(0.0, 0.0)] * percentile_90] * (max_dur_length - len(padded_video))
        gaze_padded.append(padded_video)
    # final_gaze_padded = gaze_padded + [[[(0.0, 0.0)] * max_gaze_len] * (max_fixations - len(video)) for video in gaze_locations]

    gaze_padded = torch.tensor(gaze_padded)
    gaze_mask = (gaze_padded.sum(dim=-1) != 0).float()

    return gaze_padded, gaze_mask

def get_padded_gaze(gaze_locations, max_gaze_len=8, max_dur_len=23):
    """
    Pads the gaze locations to a fixed length and returns the padded tensor and mask.
    Args:
        gaze_locations: List of gaze locations, where each gaze location is a list of tuples (x, y).
        max_gaze_len: Maximum length of gaze locations.
        max_dur_len: Maximum duration length.
    Returns:
        Padded gaze tensor and mask.
    """
    padded_all = []
    mask_all = []

    for video in gaze_locations:
        padded_video = []
        mask_video = []

        for fixation in video[:max_dur_len]:  # Limit to max_dur_len
            padded_fixation = fixation[:max_gaze_len] + [(0.0, 0.0)] * (max_gaze_len - len(fixation))
            padded_video.append(padded_fixation)

            mask = [1] * min(len(fixation), max_gaze_len) + [0] * max(0, max_gaze_len - len(fixation))
            mask_video.append(mask)
        
        # Pad missing fixations
        missing = max_dur_len - len(padded_video)
        padded_video += [[(0.0, 0.0)] * max_gaze_len] * missing
        mask_video += [[0] * max_gaze_len] * missing

        padded_all.append(padded_video)
        mask_all.append(mask_video)

    padded_gaze = torch.tensor(padded_all, dtype=torch.float32)
    gaze_mask = torch.tensor(mask_all, dtype=torch.float32)

    return padded_gaze, gaze_mask

def get_split_dataset(df_train, df_test, max_dur_length, logger):

    X = df_train["token_scan_path"].to_list()  # Already tokenized
    y = df_train["action_idx"].to_list()
    duration = df_train["duration"].to_list()   # Additional modalities
    gaze_loc = df_train["gaze"].to_list()
    test_X = df_test["token_scan_path"].to_list()  # Already tokenized
    test_y = df_test["action_idx"].to_list()
    test_dur = df_test["duration"].to_list()   # Additional modalities
    test_gaze = df_test["gaze"].to_list()

    # Split the data into training and validation sets
    # Use 10% for valdation
    train_X, val_X, train_y, val_y, train_dur, val_dur, train_gaze, val_gaze = train_test_split(
        X, y, duration, gaze_loc, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate class weights
    num_classes = len(set(train_y))
    counts = np.bincount(train_y)
    counts_safe = np.where(counts == 0, 1, counts)
    class_weights = len(train_y) / (num_classes * counts_safe)

    if "simple" or "duration" in args.output_dir:

        train_token_ids, token2id, id2token, vocab_size_train = get_token_ids(train_X, max_dur_length)       # these are the padded token ids for the scanpaths
        val_token_ids, _, _, vocab_size_val = get_token_ids(val_X, max_dur_length)
        test_token_ids, _, _, vocab_size_test = get_token_ids(test_X, max_dur_length)

        print(f"Vocab size for train: {vocab_size_train}, val: {vocab_size_val}, test: {vocab_size_test}")

        train_attention_masks = get_attention_masks(train_token_ids)          # these are the attention masks for the scanpaths
        val_attention_masks = get_attention_masks(val_token_ids)
        test_attention_masks = get_attention_masks(test_token_ids)
    
    else:
        train_token_ids = get_token_ids_2(train_X, max_dur_length)       # these are the padded token ids for the scanpaths
        val_token_ids = get_token_ids_2(val_X, max_dur_length)
        test_token_ids = get_token_ids_2(test_X, max_dur_length)
        vocab_size_train = 0  # Not used in this case

    # Get percentile 90 and 95 of gaze lengths
    percentile_90, percentile_95 = get_gaze_len(train_gaze)

    logger.info(f"Percentile 90 of gaze lengths: {percentile_90}, Percentile 95 of gaze lengths: {percentile_95}")

    train_dur_tensors = [torch.tensor(dur, dtype=torch.float32) for dur in train_dur]
    val_dur_tensors = [torch.tensor(dur, dtype=torch.float32) for dur in val_dur]
    test_dur_tensors = [torch.tensor(dur, dtype=torch.float32) for dur in test_dur]

    train_dur_padded = pad_sequence(train_dur_tensors, batch_first=True, padding_value=0.0)         # Pad to the maximum duration length
    val_dur_padded = pad_sequence(val_dur_tensors, batch_first=True, padding_value=0.0)
    test_dur_padded = pad_sequence(test_dur_tensors, batch_first=True, padding_value=0.0)

    train_dur_padded = train_dur_padded[:, :max_dur_length]  # Truncate to max_dur_length
    val_dur_padded = val_dur_padded[:, :max_dur_length]
    test_dur_padded = test_dur_padded[:, :max_dur_length]

    train_duration_mask = (train_dur_padded != 0).float()
    val_duration_mask = (val_dur_padded != 0).float()
    test_duration_mask = (test_dur_padded != 0).float()

    train_gaze_padded, train_gaze_mask = get_padded_gaze(train_gaze, percentile_90, max_dur_length)
    val_gaze_padded, val_gaze_mask = get_padded_gaze(val_gaze, percentile_90, max_dur_length)
    test_gaze_padded, test_gaze_mask = get_padded_gaze(test_gaze, percentile_90, max_dur_length)

    train_dataset = {
        "input_ids": torch.tensor(train_token_ids),
        "attention_mask": torch.tensor(train_attention_masks),
        "labels": torch.tensor(train_y),
        "duration": train_dur_padded,
        "gaze": train_gaze_padded,
        "duration_mask": train_duration_mask,
        "gaze_mask": train_gaze_mask,
    }

    val_dataset = {
        "input_ids": torch.tensor(val_token_ids),
        "attention_mask": torch.tensor(val_attention_masks),
        "labels": torch.tensor(val_y),
        "duration": val_dur_padded,
        "gaze": val_gaze_padded,
        "duration_mask": val_duration_mask,
        "gaze_mask": val_gaze_mask,
    }

    test_dataset = {
        "input_ids": torch.tensor(test_token_ids),
        "attention_mask": torch.tensor(test_attention_masks),
        "labels": torch.tensor(test_y),
        "duration": test_dur_padded,
        "gaze": test_gaze_padded,
        "duration_mask": test_duration_mask,
        "gaze_mask": test_gaze_mask,
    }

    return train_dataset, val_dataset, test_dataset, test_X, class_weights, vocab_size_train

class GazeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.device = DEVICE

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, idx):
        return {key: value[idx].clone().detach() for key, value in self.dataset.items()}

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch["labels"]

        if args.wo_fixation:
            # Remove gaze and gaze_mask if not using fixation data
            model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
        elif "simple" in args.output_dir:
            model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
        elif "duration" in args.output_dir or "distilbert" in args.output_dir:
            model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels", "duration", "duration_mask"]}

        optimizer.zero_grad()

        # Forward pass
        outputs = model(**model_inputs)
        logits = outputs['logits']

        loss = outputs['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(logits, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def evaluate(model, dataloader, logger):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    logger.info("Evaluating...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch["labels"]

            if args.wo_fixation:
                # Remove gaze and gaze_mask if not using fixation data
                model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
            elif "simple" in args.output_dir:
                model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
            elif "duration" in args.output_dir or "distilbert" in args.output_dir:
                model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels", "duration", "duration_mask"]}

            outputs = model(**model_inputs)
            logits = outputs["logits"]

            # loss = criterion(outputs, labels)

            loss = outputs["loss"]

            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    # print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def eval_test(model, test_dataset, id2label, save_path, data_split, logger):
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    total_loss = 0
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}

    all_preds = []
    all_labels = []

    logger.info(f"Evaluating on {data_split} dataset...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Evaluating {data_split}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch["labels"]

            if args.wo_fixation:
                # Remove gaze and gaze_mask if not using fixation data
                model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
            elif "simple" in args.output_dir:
                model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
            elif "duration" in args.output_dir or "distilbert" in args.output_dir:
                model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels", "duration", "duration_mask"]}

            outputs = model(**model_inputs)
            logits = outputs["logits"]

            loss = outputs["loss"]

            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels, predicted):
                t = t.item()
                p = p.item()
                per_class_correct[t] = per_class_correct.get(t, 0) + int(t == p)
                per_class_total[t] = per_class_total.get(t, 0) + 1
                all_preds.append(p)
                all_labels.append(t)
        
    avg_loss = total_loss / len(test_dataloader)
    avg_accuracy = correct / total

    per_class_acc = {}
    class_accuracies = []

    for cls_id in sorted(per_class_total.keys()):
        correct_count = per_class_correct.get(cls_id, 0)
        total_count = per_class_total[cls_id]
        acc = correct_count / total_count
        per_class_acc[cls_id] = acc
        class_accuracies.append(acc)

    mean_class_acc = sum(class_accuracies) / len(class_accuracies)

    logger.info(f"Validation Loss: {avg_loss:.4f}, Overall Accuracy: {avg_accuracy:.4f}, Mean Class Accuracy: {mean_class_acc:.4f}")

    # Save per-class accuracies to CSV
    class_stats = []
    for cls_id in per_class_acc:
            class_stats.append({
                "class_id": cls_id,
                "class_name": id2label[cls_id] if id2label else str(cls_id),
                "accuracy": per_class_acc[cls_id],
                "correct": per_class_correct[cls_id],
                "total": per_class_total[cls_id]
            })
    df_stats = pd.DataFrame(class_stats)
    df_stats = df_stats.sort_values(by="class_id")
    csv_path = os.path.join(save_path, f"per_class_accuracy_{data_split}.csv")
    df_stats.to_csv(csv_path, index=False)

    logger.info(f"Per-class accuracy saved to: {csv_path}")

    if data_split.lower() == "test":
        class_names = [id2label[i] for i in sorted(per_class_correct.keys())]
        num_classes = len(class_names)
        # Save confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=sorted(per_class_correct.keys()))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2label.values()))
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(cm, annot=True, fmt='.2f', ax=ax, cmap='Blues', cbar=False,
                   xticklabels=range(num_classes), yticklabels=range(num_classes),
                   annot_kws={"size": 10}, square=True)
        # disp.plot(ax=ax, xticks_rotation=45, cmap=plt.cm.Blues, colorbar=False)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        cm_path = os.path.join(save_path, f"confusion_matrix_{data_split}.png")
        plt.title(f"Confusion Matrix for {data_split} dataset", fontsize=16)

        # Create custom legend
        legend_elements = [
            Patch(facecolor='none', edgecolor='none', label=f"{i}: {name}")
            for i, name in enumerate(class_names)
        ]
        plt.legend(handles=legend_elements, title="Class Index → Name",
                loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, title_fontsize=12)

        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close(fig)
        logger.info(f"Confusion matrix saved to: {cm_path}")

    return avg_loss, avg_accuracy, mean_class_acc, per_class_acc

def plot_data_distribution(dataset, id2label, data_split):
    """
    Plot the distribution of action classes in the dataset.
    """
    # Extract labels from the dataset
    labels = dataset["labels"].tolist()
    label_series = pd.Series(labels)

    # Count occurrences of each label
    class_counts = label_series.value_counts().sort_index()
    
    # Map index to class names (if needed)
    class_names = [id2label[idx] for idx in class_counts.index]

    save_path = os.path.join(args.root_path, "class_distribution")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts.values, color='skyblue')
    plt.xlabel('Action Classes')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Action Classes') 
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"data_split_{data_split}.png"))

    # Also save the class counts to a CSV file
    class_counts_df = pd.DataFrame({
        "class_id": class_counts.index,
        "class_name": class_names,
        "count": class_counts.values
    })
    class_counts_df.to_csv(os.path.join(save_path, f"class_counts_{data_split}.csv"), index=False)

def train_split(df_train, 
                id2label, 
                label2id, 
                df_test,
                output_dir, action_classes, max_length, logger, exp_name):
    """
    Train the model on the given split of the dataset.
    """
    logger.info("Loading data and setting up the model...")
    best_model = None
    start_time = time.time()

    train_loss_all, valid_loss_all = [], []
    train_acc_all, valid_acc_all = [], []

    train_dataset, val_dataset, test_dataset, test_X, class_weights, vocab_size = get_split_dataset(df_train, df_test, max_length, logger)

    # Plot the data distribution
    # plot_data_distribution(train_dataset, id2label, "train")
    # plot_data_distribution(test_dataset, id2label, "test")

    train_dataset = GazeDataset(train_dataset)
    val_dataset = GazeDataset(val_dataset)
    test_dataset = GazeDataset(test_dataset)

    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}")


    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True,)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False,)

    num_labels = len(id2label)
    logger.info(f"Number of labels: {num_labels}, Class weights: {class_weights}")

    if args.wo_fixation:
        logger.info("Training without fixation data using DistilBERT.")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    elif "simple" in args.output_dir:
        logger.info("Training with simple scanpath classifier.")

        logger.info(f"Vocab size: {vocab_size}")
        pad_idx = 0  # Assuming 0 is the padding index

        model = SimpleActionClassifier(vocab_size=vocab_size,
                                         embed_dim=args.embed_dim,
                                         hidden_dim=args.hidden_dim,
                                         pad_idx=pad_idx,
                                         class_weights=class_weights,
                                         num_classes=num_labels)
        logger.info(f"Model: {model}")
    elif "duration" in args.output_dir:
        logger.info("Training with duration data using RNN.")
        model = SimpleDurationActionClassifier(vocab_size=vocab_size,
                                               embed_dim=args.embed_dim,
                                               hidden_dim=args.hidden_dim,
                                               pad_idx=0,  # Assuming 0 is the padding index
                                               class_weights=class_weights,
                                               num_classes=num_labels)
        logger.info(f"Model: {model}")
    elif "distilbert" in args.output_dir:
        logger.info("Training with fixation data using DistilBERT.")
        model = ActionClassifier(hidden_dim=args.hidden_dim,
                                num_labels=num_labels,
                                class_weights=class_weights)
        logger.info(f"Model: {model}")
    
    # Set device to CUDA if available
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_built():
        device = torch.device("mps")

    # Ensure the model is on the same device
    model.to(device)
    logger.info(f"Model loaded on {next(model.parameters()).device}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # criterion = nn.CrossEntropyLoss()

    num_epochs = args.num_epochs

    logger.info(f"Optimizer: {optimizer}, Number of epochs: {num_epochs}")
    logger.info("Starting training...")

    # Save the best model
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        
        train_loss, train_acc = train_one_epoch(model, train_dataloader, optimizer)
        val_loss, val_acc = evaluate(model, val_dataloader, logger)

        train_loss_all.append(train_loss)
        valid_loss_all.append(val_loss)
        train_acc_all.append(train_acc)
        valid_acc_all.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save only the best model
        best_model_path = os.path.join(exp_name, "best_model.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_model_path)
            best_model = model
            logger.info(f"Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")
    
    save_plots(train_loss_all, valid_loss_all, train_acc_all, valid_acc_all, exp_name)

    end_time = time.time()

    logger.info("Training complete.")
    logger.info(f"Total training time: {end_time - start_time:.2f} seconds")

    return train_dataset, val_dataset, test_dataset, best_model

def save_plots(train_loss_all, valid_loss_all, train_acc_all, valid_acc_all, exp_name):
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc_all, color="green", linestyle="-", 
        label="train accuracy"
    )
    plt.plot(
        valid_acc_all, color="blue", linestyle="-", 
        label="validation accuracy"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(exp_name, "accuracy_plot.png"))
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss_all, color="orange", linestyle="-", 
        label="train loss"
    )
    plt.plot(
        valid_loss_all, color="red", linestyle="-", 
        label="validation loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(exp_name, "loss_plot.png"))


#####################################################
#                      Training                     #
#####################################################

def main(args):
    set_seed(seed=args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.output_dir}_{timestamp}"

    df_train, id2label, label2id, action_classes_train, df_test, action_classes_test = load_data(args.root_path)
    logger = create_logger(log_dir=exp_name)
    logger.info(f"Data loaded from {args.root_path}.")
    logger.info(f"Maximum length of tokenized scan paths: {args.max_length}")
    logger.info(f"Seed set to {args.seed}.")
    train_dataset, val_dataset, test_dataset, best_model = train_split(df_train, 
                                    id2label, 
                                    label2id, 
                                    df_test,
                                    args.output_dir, action_classes_train, args.max_length, logger, exp_name)
    

    # Evaluate the best model on the test set
    logger.info("Evaluating the best model on the test set...")
    _, _, _, _ = eval_test(best_model, train_dataset, id2label, exp_name, "train", logger)
    _, _, _, _ = eval_test(best_model, val_dataset, id2label, exp_name, "val", logger)
    avg_loss, avg_acc, mean_class_acc, per_class_acc = eval_test(best_model, test_dataset, id2label, exp_name, "test", logger)
    logger.info("Training and evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--root_path', type=str, default='/Users/xiaoxiao/Documents/Studium/Lab/GazeEstimation/Results/scanpaths')
    parser.add_argument('--root_path', type=str, default=r"C:\Users\Su\Documents\Codes\Results\EGTEA\experiment_3")
    # parser.add_argument('--output_dir', type=str, default='/Users/xiaoxiao/Documents/Studium/Lab/GazeEstimation/Results/final/23_test_simple')
    parser.add_argument('--output_dir', type=str, default=r"C:\Users\Su\Documents\Codes\Results\EGTEA\final\all\simple")
    parser.add_argument('--max_length', type=int, default=7, help='Maximum length of the tokenized scan paths')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs for training')
    parser.add_argument('--wo_fixation', action='store_true', help='Whether to train without fixation data')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension for the model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args)