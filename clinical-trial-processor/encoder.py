import os
import torch
import torch.nn as nn
from torchcrf import CRF
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_from_disk, ClassLabel, Sequence

from utils import collate_fn
from ncbi_disease_dataset import NCBIDataset
from constants import NCBI_DATASET_VOCAB_KEYS, NCBI_DATASET_DATA_FIELDS, NCBI_DATASET_TAG_TO_IX, LOCAL_NCBI_DATASET_DISK_PATH, NCBI_DATASET_NEGATION_TRIGGERS, NCBI_DATASET_NEGATION_WINDOW_SIZE, MODEL_PARAMS

class ClinicalTrialEncoder(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(ClinicalTrialEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        
        # Embedding Layer: Converts token IDs to dense vectors
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM Layer: Reads sequence forward and backward
        # batch_first=True makes input/output tensors shape (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        
        # Linear Layer: Maps BiLSTM outputs to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
        # CRF Layer: Enforces sequence transition rules
        # batch_first=True ensures compatibility with our LSTM output
        self.crf = CRF(tagset_size, batch_first=True)

    def _get_lstm_features(self, sentence_batch, mask):
        # Pass through embeddings
        embeds = self.word_embeds(sentence_batch)
        
        # Pass through BiLSTM
        lstm_out, _ = self.lstm(embeds)
        
        # Map to tag space (these are Emission Scores)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence_batch, tags_batch, mask):
        # This is the loss function used during TRAINING
        lstm_feats = self._get_lstm_features(sentence_batch, mask)
        
        # The CRF computes the negative log likelihood loss
        # We return the negative because PyTorch optimizers minimize loss
        loss = -self.crf(lstm_feats, tags_batch, mask=mask, reduction='mean')
        return loss

    def decode(self, sentence_batch, mask):
        # This is the Viterbi decoding used during INFERENCE
        lstm_feats = self._get_lstm_features(sentence_batch, mask)
        
        # Returns the most likely sequence of tags
        best_tag_sequence = self.crf.decode(lstm_feats, mask=mask)
        return best_tag_sequence
    
class ClinicalTrialEncoderTrainer():
    def __init__(self):        
        # Define vocabulary of tags
        self.tag_to_ix = NCBI_DATASET_TAG_TO_IX

        # Reverse mapping for Inference (Decoding)
        self.ix_to_tag = {v: k for k, v in self.tag_to_ix.items()}
        
        self.word_to_ix = {NCBI_DATASET_VOCAB_KEYS.PADDING.value: 0, NCBI_DATASET_VOCAB_KEYS.UNKNOWN.value: 1} # Reserve 0 for padding, 1 for unknown words
        self.load_dataset_from_disk()
        self.build_vocabulary()
    
    def load_dataset_from_disk(self):
        """
        Load the original dataset and made changes to allow for extra classes
        """
        # Load the dataset
        print("Loading raw NCBI dataset...")
        local_dataset_path = os.path.abspath(LOCAL_NCBI_DATASET_DISK_PATH)
        raw_dataset = load_from_disk(local_dataset_path)
        
        # Fetch the features
        new_features = raw_dataset['train'].features.copy()
        
        # Extract tag names in order of Ids
        tag_names = [k for k,v in sorted(self.tag_to_ix.items(), key=lambda item: item[1])]
        
        # Construct list of features
        new_features[NCBI_DATASET_DATA_FIELDS.NER_TAGS.value] = Sequence(
            feature = ClassLabel(num_classes=len(self.tag_to_ix), names=tag_names)
        )
        
        # Inject new classes
        self.dataset = raw_dataset.map(self._inject_negation, features=new_features)
        
        # Filter out empty sequences
        self.dataset = self.dataset.filter(lambda example: len(example[NCBI_DATASET_DATA_FIELDS.TOKENS.value]) > 0)
        
    def _inject_negation(self, example):
        """
        Add support for negation tags
        """
        tokens = [t.lower() for t in example[NCBI_DATASET_DATA_FIELDS.TOKENS.value]]
        tags = example[NCBI_DATASET_DATA_FIELDS.NER_TAGS.value].copy()
        
        for i, tag in enumerate(tags):
            if tag == self.tag_to_ix["B-Disease"]:
                start_window = max(0, i - NCBI_DATASET_NEGATION_WINDOW_SIZE)
                window_tokens = tokens[start_window:i]
                
                if any(trigger in window_tokens for trigger in NCBI_DATASET_NEGATION_TRIGGERS):
                    # Flip B-Disease to B-Neg-Disease
                    tags[i] = self.tag_to_ix['B-Neg-Disease']
                    
                    # Change the closing tag
                    j = i + 1
                    while j < len(tags) and tags[j] == self.tag_to_ix['I-Disease']:
                        tags[j] = self.tag_to_ix['I-Neg-Disease']
                        j += 1
            
        return {NCBI_DATASET_DATA_FIELDS.NER_TAGS.value: tags} # return the mutated tag
    
    def build_vocabulary(self):
        for split in self.dataset.keys():
            for example in self.dataset[split]:
                for word in example[NCBI_DATASET_DATA_FIELDS.TOKENS.value]:
                    if word not in self.word_to_ix:
                        self.word_to_ix[word] = len(self.word_to_ix)

        print(f"Total vocabulary size: {len(self.word_to_ix)}")
    
    def train(self):
        train_data = NCBIDataset(self.dataset['train'], self.word_to_ix)
        
        # Create the data loader
        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Init the model
        model = ClinicalTrialEncoder(
            vocab_size=len(self.word_to_ix), 
            tagset_size=len(self.tag_to_ix), # This tells the model there are 9 possible classes
            embedding_dim=MODEL_PARAMS.EMBEDDING_DIM.value, 
            hidden_dim=MODEL_PARAMS.HIDDEN_DIM.value
        )

        optimizer = optim.Adam(model.parameters(), lr=MODEL_PARAMS.LR.value)

        for epoch in range(int(MODEL_PARAMS.EPOCHS.value)):
            model.train()
            total_loss = 0
            
            for sentences, tags, masks in train_loader:
                # Clear gradients
                model.zero_grad()
                
                # Forward pass to get loss
                loss = model(sentences, tags, mask=masks)
                
                # Backpropagation
                loss.backward()
                
                # Update weights
                optimizer.step()
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader)}")

        save_dir = os.path.abspath(MODEL_PARAMS.WEIGHTS_SAVE_DIR.value)
        os.makedirs(save_dir, exist_ok=True)
        
        # Assemble everything
        content = {
            'model_state_dict': model.state_dict(),
            'word_to_ix': self.word_to_ix,
            'tag_to_ix': self.tag_to_ix
        }
        
        save_path = os.path.join(save_dir, MODEL_PARAMS.WEIGHTS_NAME.value)
        torch.save(content, save_path)
        
        print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    trainer = ClinicalTrialEncoderTrainer()
    trainer.train()