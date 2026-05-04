from datasets import load_dataset, load_from_disk, ClassLabel, Sequence
import torch
import os
from torch.utils.data import Dataset
from utils import clean_dataset_name

from clinical_trial_processor.constants import BC5CDR_DATASET_DATA_FIELDS, BC5CDR_DATASET_NATIVE_TAG_TO_IX, BC5CDR_DATASET_TAG_TO_IX, DATASET_DISK_PATH, DATASET_VOCAB_KEYS, NCBI_DATASET_DATA_FIELDS, NCBI_DATASET_NAME, BC5CDR_DATASET_NAME, NCBI_DATASET_TAG_TO_IX, DATASET_NEGATION_TRIGGERS, DATASET_NEGATION_WINDOW_SIZE

class BIOTaggingDataset():
    """Dataset used for BIO Tagging task"""
    
    class TorchDataset(Dataset):
        def __init__(self, hf_dataset, word_to_ix, dataFields):
            self.data = hf_dataset
            self.word_to_ix = word_to_ix
            self.dataFields = dataFields

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            example = self.data[idx]
            
            # Convert words to integers
            token_ids = [self.word_to_ix.get(word, self.word_to_ix[DATASET_VOCAB_KEYS.UNKNOWN.value]) for word in example[self.dataFields.TOKENS.value]]
            tag_ids = example[self.dataFields.NER_TAGS.value]
            
            return torch.tensor(token_ids), torch.tensor(tag_ids)
    
    def __init__(self, datasetName):
        self.datasetName = datasetName
        self.cleanedDatasetName = clean_dataset_name(self.datasetName)
        self.local_dataset_path = os.path.abspath(os.path.join(DATASET_DISK_PATH, self.cleanedDatasetName))
        
        if not os.path.exists(self.local_dataset_path):
            self.download()
        
        self.wordToIx = {DATASET_VOCAB_KEYS.PADDING.value: 0, DATASET_VOCAB_KEYS.UNKNOWN.value: 1}
        self.setupDataset()
        self.load_dataset_from_disk()
        self.build_vocabulary()
        self.trainData = self.TorchDataset(self.dataset['train'], self.wordToIx, self.dataFields)
        self.testData = self.TorchDataset(self.dataset['test'], self.wordToIx, self.dataFields)
    
    def download(self):
        print("Downloading and loading BIO Tagging disease/medication dataset...")
        dataset = load_dataset(self.datasetName, trust_remote_code=True)

        dataset.save_to_disk(self.local_dataset_path)

        print(f"Dataset loaded successfully and saved for offline use at: {self.local_dataset_path}!")
        print("Splits:", dataset.keys())

        # Print the first training example to verify
        print("\nFirst training record:")
        print(dataset['train'][0])
    
    def setupDataset(self):
        if self.datasetName == BC5CDR_DATASET_NAME:
            self.tagToIx = BC5CDR_DATASET_TAG_TO_IX
            self.dataFields = BC5CDR_DATASET_DATA_FIELDS
        else:
            # Default to NCBI
            self.tagToIx = NCBI_DATASET_TAG_TO_IX
            self.dataFields = NCBI_DATASET_DATA_FIELDS
        
        self.ixToTag = {v: k for k, v in self.tagToIx.items()}
    
    def load_dataset_from_disk(self):
        """
        Load the original dataset and made changes to allow for extra classes
        """
        # Load the dataset
        print(f"Loading raw {self.datasetName} dataset...")
        raw_dataset = load_from_disk(self.local_dataset_path)
        
        # Fetch the features
        new_features = raw_dataset['train'].features.copy()
        
        # Extract tag names in order of Ids
        tag_names = [k for k,v in sorted(self.tagToIx.items(), key=lambda item: item[1])]
        
        # Construct list of features
        new_features[self.dataFields.NER_TAGS.value] = Sequence(
            feature = ClassLabel(num_classes=len(self.tagToIx), names=tag_names)
        )
        
        # Inject new classes
        self.dataset = raw_dataset.map(self._inject_negation, features=new_features)
        
        # Filter out empty sequences
        self.dataset = self.dataset.filter(lambda example: len(example[self.dataFields.TOKENS.value]) > 0)
    
    def _translate_tags(self, tags, native_tags):
        new_tags = []
        
        for raw_tag_id in tags:
            tag_string = native_tags[raw_tag_id]
            new_tags.append(self.tagToIx[tag_string])
        
        return new_tags

    def _inject_negation(self, example):
        """
        Add support for negation tags
        """
        tokens = [t.lower() for t in example[self.dataFields.TOKENS.value]]
        tags = example[self.dataFields.NER_TAGS.value].copy()
        
        if self.datasetName == BC5CDR_DATASET_NAME:
            tags = self._translate_tags(tags, BC5CDR_DATASET_NATIVE_TAG_TO_IX)
        
        for i, tag in enumerate(tags):
            if tag in (self.tagToIx["B-Disease"], self.tagToIx['B-Chemical']):
                negated_tag = 'B-Neg-Disease' if tag == self.tagToIx["B-Disease"] else 'B-Neg-Chemical'
                closing_start_tag = 'I-Disease' if tag == self.tagToIx["B-Disease"] else 'I-Chemical'
                closing_end_tag = 'I-Neg-Disease' if tag == self.tagToIx["B-Disease"] else "I-Neg-Chemical"
                
                start_window = max(0, i - DATASET_NEGATION_WINDOW_SIZE)
                window_tokens = tokens[start_window:i]
                
                if any(trigger in window_tokens for trigger in DATASET_NEGATION_TRIGGERS):
                    # Flip B-Disease to B-Neg-Disease
                    tags[i] = self.tagToIx[negated_tag]
                        
                    # Change the closing tag
                    j = i + 1
                    while j < len(tags) and tags[j] == self.tagToIx[closing_start_tag]:
                        tags[j] = self.tagToIx[closing_end_tag]
                        j += 1
            
        return {self.dataFields.NER_TAGS.value: tags} # return the mutated tag
    
    def build_vocabulary(self):
        for split in self.dataset.keys():
            for example in self.dataset[split]:
                for word in example[self.dataFields.TOKENS.value]:
                    if word not in self.wordToIx:
                        self.wordToIx[word] = len(self.wordToIx)

        print(f"Total vocabulary size: {len(self.wordToIx)}")
    