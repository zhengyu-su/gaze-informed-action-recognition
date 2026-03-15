from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel
import torch
import torch.nn as nn

class ActionClassifier(nn.Module):
    def __init__(self, hidden_dim, num_labels, class_weights=None, max_duration_len=107, max_gaze_len=57):
        super().__init__()
        self.num_labels = num_labels
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # === Freeze all BERT layers ===
        for param in self.bert.parameters():
            param.requires_grad = False

        # === Unfreeze the last layer of BERT ===
        for param in self.bert.transformer.layer[-1].parameters():
            param.requires_grad = True

        # self.gaze_proj = nn.Linear(2, hidden_dim)          # 2D gaze data (x, y)
        # self.duration_proj = nn.Linear(max_duration_len, hidden_dim)
        # self.gaze_proj = nn.Linear(2, hidden_dim)
        self.duration_proj = nn.Linear(1, hidden_dim) 
        self.duration_to_bert = nn.Linear(hidden_dim, 768)     # 1D duration data
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(768 * 2, self.num_labels)   # 3 for bert_embed, gaze_embed, duration_embed, 2 for bert_embed and duration_embed

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
    
    def forward(self, input_ids, attention_mask, duration, duration_mask, labels=None):
        '''
        duration: (batch, max_duration_len)  duration of each fixation in one video clip
        duration_mask: (batch, max_duration_len)
        gaze: (batch, max_duration_len, max_gaze_len, 2)
        gaze_mask: (batch, max_duration_len, max_gaze_len)
        '''
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_built():
            device = torch.device("mps")

        # BERT encoding
        bert_output = self.bert(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        cls_embed = bert_output.last_hidden_state[:, 0, :]  # Shape: (batch, 768)

        cls_embed = self.dropout(cls_embed)  # Apply dropout to CLS token embedding
        
        # Duration projection and masking
        duration_embed = self.duration_proj(duration.unsqueeze(-1))  # Shape: (batch, max_duration_len, hidden_dim)
        duration_embed = duration_embed * duration_mask.to(device).unsqueeze(-1)  #
        duration_embed = (duration_embed.sum(dim=1) / duration_mask.to(device).sum(dim=1, keepdim=True)).nan_to_num(0)
        duration_embed = self.duration_to_bert(duration_embed)  # Shape: (batch, 768)

        duration_embed = self.dropout(duration_embed)  # Apply dropout to duration embedding

        # # bert_embeds = bert_output.last_hidden_state  # Shape: (batch, seq
        # token_embeddings = bert_output.last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        # mask = attention_mask.to(device).unsqueeze(-1).float()  # Shape: (batch, seq_len, 1)
        # # bert_embeds = bert_output.last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        # # bert_embed = bert_output.last_hidden_state[:, 0, :]
        # bert_embed = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)  # Shape: (batch, hidden_size)
        # # logits = bert_output.logits

        # duration = duration.to(device)  # Shape: (batch, max_duration_len)
        # duration = (duration - duration.mean()) / duration.std()

        # duration_embed = self.duration_proj(duration.unsqueeze(-1)) # Shape: (batch, 1, hidden_dim)
        # duration_embed = duration_embed * duration_mask.to(device).unsqueeze(-1)
        # duration_embed = (duration_embed.sum(dim=1) / duration_mask.to(device).sum(dim=1, keepdim=True)).nan_to_num(0)

        # gaze = gaze.reshape(batch_size, max_duration_len, max_gaze_len * 2)
        # gaze_embed = self.gaze_proj(gaze.to(device))
        # gaze_embed = gaze_embed * gaze_mask.to(device).unsqueeze(-1)

        combined = torch.cat([cls_embed, duration_embed], dim=1)  # Shape: (batch, 768 * 2)

        combined = self.dropout(combined)  # Apply dropout to combined embeddings

        logits = self.classifier(combined)  # Shape: (batch, num_labels)

        # Compute loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"logits": logits, "loss": loss} if labels is not None else {"logits": logits}


class SimpleActionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_classes=106, class_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.7)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32), label_smoothing=0.1)
            print(f"Using class weights")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("No class weights provided, using default CrossEntropyLoss")

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # output, (hn, cn) = self.lstm(packed)
        # hn_cat = torch.cat((hn[0], hn[1]), dim=1)  # Concatenate forward and backward final hidden states
        # hn_cat = self.dropout(hn_cat)  # Apply dropout
        # hn_cat = self.layer_norm(hn_cat)  # Apply layer normalization

        output, hidden = self.rnn(packed)

        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate forward and backward final hidden states
        final_hidden = self.dropout(final_hidden)  # Apply dropout
    
        logits = self.classifier(final_hidden)  # Shape: (batch_size, num_classes)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"logits": logits, "loss": loss} if labels is not None else {"logits": logits}


'''
# Example Usage
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokens = ["egg", "plate", "plate", "egg", "knife"]
inputs = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)

gaze = torch.randint(0, 10, (1, len(tokens)))  # Example gaze data
fixation = torch.randint(0, 10, (1, len(tokens)))  # Example fixation data

# model = ActionClassifier(gaze_bins=10, fixation_bins=10, hidden_dim=64)
model = ActionClassifier(hidden_dim=64,)
output = model(inputs["input_ids"], inputs["attention_mask"], gaze, fixation)
print(output.shape)  # (batch_size, num_classes)
'''

class SimpleDurationActionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_classes=106, class_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.duration_proj = nn.Linear(1, hidden_dim * 2)  # Match GRU hidden output size

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2 * 2, num_classes)  # GRU output + duration proj

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, duration, duration_mask, labels=None):
        x = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        _, hidden = self.rnn(packed)  # hidden: (2, batch, hidden_dim)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_dim*2)

        # Project durations: (batch, seq_len, 1) → (batch, seq_len, hidden_dim*2)
        duration = duration.unsqueeze(-1).float()
        duration_embed = self.duration_proj(duration)

        # Apply duration mask and aggregate
        duration_mask = duration_mask.unsqueeze(-1)
        duration_embed = duration_embed * duration_mask
        duration_embed = duration_embed.sum(dim=1) / (duration_mask.sum(dim=1) + 1e-6)

        # Combine GRU + duration
        combined = torch.cat([final_hidden, duration_embed], dim=1)  # (batch, hidden_dim*4)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"logits": logits, "loss": loss} if labels is not None else {"logits": logits}