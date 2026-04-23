import torch
from torch.utils.data import Dataset

from constants import NCBI_DATASET_DATA_FIELDS, NCBI_DATASET_VOCAB_KEYS

class NCBIDataset(Dataset):
    """
    This dataset contains the disease name and concept annotations of the NCBI disease corpus, 
    a collection of 793 PubMed abstracts fully annotated at the mention and 
    concept level to serve as a research resource for the biomedical natural language processing community.
    """

    def __init__(self, hf_dataset, word_to_ix):
        self.data = hf_dataset
        self.word_to_ix = word_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Data Fields:
            id: Sentence identifier.
            tokens: Array of tokens composing a sentence.
            ner_tags: Array of tags, where 0 indicates no disease mentioned, 1 signals the first token 
            of a disease and 2 the subsequent disease tokens.
        """
        
        example = self.data[idx]
        
        # Convert words to integers
        token_ids = [self.word_to_ix.get(word, self.word_to_ix[NCBI_DATASET_VOCAB_KEYS.UNKNOWN]) for word in example[NCBI_DATASET_DATA_FIELDS.TOKENS]]
        tag_ids = example[NCBI_DATASET_DATA_FIELDS.NER_TAGS]
        
        return torch.tensor(token_ids), torch.tensor(tag_ids)