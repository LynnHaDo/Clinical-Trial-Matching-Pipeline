import os
import torch
import torch.nn as nn
from torchcrf import CRF
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import collate_fn
from constants import MODEL_PARAMS, DEFAULT_DATASET
from dataset import BIOTaggingDataset

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
    def __init__(self, datasetName):       
        self.datasetName = datasetName 
        self.dataset = BIOTaggingDataset(datasetName)
        
        # Define vocabulary of tags
        self.tag_to_ix = self.dataset.tagToIx

        # Reverse mapping for Inference (Decoding)
        self.ix_to_tag = self.dataset.ixToTag
        
        self.word_to_ix = self.dataset.wordToIx
        self.dataFields = self.dataset.dataFields
        self.trainData = self.dataset.trainData
    
    def train(self):
        train_data = self.trainData
        
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

        save_dir = os.path.abspath(os.path.join(MODEL_PARAMS.WEIGHTS_SAVE_DIR.value, self.dataset.cleanedDatasetName))
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
    trainer = ClinicalTrialEncoderTrainer(DEFAULT_DATASET)
    trainer.train()