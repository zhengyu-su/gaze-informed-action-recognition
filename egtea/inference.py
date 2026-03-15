import os
import sys
import argparse
from classification import load_data, get_dataset, GazeDataset, custom_collate_fn
from torch.utils.data import DataLoader
import torch
from safetensors.torch import load_file

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.sequence_models import ActionClassifier

def main(args):
    device = torch.device("mps" if torch.has_mps else "cpu")
    data, id2label, label2id, action_classes = load_data(args.root_path)
    train_dataset, val_dataset, test_dataset, test_X = get_dataset(data, args.max_length)

    model_dir = os.path.join(args.output_dir, "best_model/model.safetensors")
    num_labels = len(id2label)
    model = ActionClassifier(hidden_dim=64,
                             num_labels=num_labels).to(device)
    model_weights = load_file(model_dir)
    model.load_state_dict(model_weights)

    val_dataset = GazeDataset(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate_fn)

    # For multi-modal
    # Iterate over the validation dataset (val_dataloader)
    n_correct = 0
    n_samples = len(val_dataset)
    for i, batch in enumerate(val_dataloader):
        print(f"{i}th data: \n")
        true_class_id = int(batch["labels"])
        with torch.no_grad():
            # Ensure the batch is in the correct device (if needed)
            batch = {k: v.to(device) for k, v in batch.items()}  # Ensure batch is on the same device as model

            res = model(**batch)  # Forward pass
            logits = res["logits"]  # Now you can unpack the batch correctly
            predicted_class_id = logits.argmax().item()
            print(f"Predicted_class: {id2label[predicted_class_id]}, ")
            print(f"True class: {id2label[true_class_id]} \n")
            if predicted_class_id == true_class_id:
                n_correct += 1

            loss = res["loss"] 
    
    print(f"{n_correct}/{n_samples} correctly classified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='/Users/xiaoxiao/Documents/Studium/Lab/GazeEstimation/Results/EGTEAGaze+/experiment_5')
    parser.add_argument("--output_dir", type=str, default='/Users/xiaoxiao/Documents/Studium/Lab/GazeEstimation/Results/EGTEAGaze+/experiment_5/classifier_results')
    parser.add_argument("--max_length", type=int, default=150)
    args = parser.parse_args()
    main(args)