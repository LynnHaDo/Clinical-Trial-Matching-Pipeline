import torch
import torch.nn as nn
from torchcrf import CRF
import torch.optim as optim

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
        self.tag_to_ix = {
                "O": 0,
                "B-Disease": 1,
                "I-Disease": 2,
                "B-Chemical": 3,
                "I-Chemical": 4,
                "B-Neg-Disease": 5,
                "I-Neg-Disease": 6,
                "<START_TAG>": 7, # Required by CRF to know where a sequence begins
                "<STOP_TAG>": 8   # Required by CRF to know where a sequence ends
        }

        # Reverse mapping for Inference (Decoding)
        self.ix_to_tag = {v: k for k, v in self.tag_to_ix.items()}
    
    def train(self):
        model = ClinicalTrialEncoder(
            vocab_size=len(word_to_ix), 
            tagset_size=len(self.tag_to_ix), # This tells the model there are 9 possible classes
            embedding_dim=100, 
            hidden_dim=256
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 30

        for epoch in range(epochs):
            model.train()
            for sentences, tags, masks in dataloader:
                # Clear gradients
                model.zero_grad()
                
                # Forward pass to get loss
                loss = model(sentences, tags, mask=masks)
                
                # Backpropagation
                loss.backward()
                
                # Update weights
                optimizer.step()
    
        model.eval()
        with torch.no_grad():
            # decode() runs the Viterbi algorithm to find the absolute best path
            predicted_tags = model.decode(new_patient_sentences, mask=new_masks)

        # predicted_tags might look like: [0, 0, 0, 3, 4, 0] 
        # which translates back to: ['O', 'O', 'O', 'B-Neg-Disease', 'I-Neg-Disease', 'O']