import torch
from torch.nn.utils.rnn import pad_sequence
from constants import NCBI_DATASET_VOCAB_KEYS

def collate_fn(batch):
    """
    Make sure that sentences are in the same length (i.e. pad sentences of shorter length)
    """
    
    sentences = [item[0] for item in batch]
    tags = [item[1] for item in batch]
        
    # Pad sequences with 0s
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0) # 0 is the 'O' tag
        
    # Create the mask: 1 for real words, 0 for padding
    mask = (padded_sentences != 0).bool()
        
    return padded_sentences, padded_tags, mask

def prepare_sequence(seq, to_ix):
    """
    Converts a text string into a tensor of vocabulary IDs.
    """
    # Split text into tokens (in a real pipeline, use a proper tokenizer like spacy)
    tokens = str(seq).split() 
    idxs = [to_ix.get(w.lower(), to_ix[NCBI_DATASET_VOCAB_KEYS.UNKNOWN.value]) for w in tokens]
    return torch.tensor(idxs, dtype=torch.long)

def extract_entities(text, tags):
    """
    Parses tags back into actual phrases
    """
    tokens = str(text).split()
    entities = []
    current_entity = []
    current_tag_type = None

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            # Save previous entity if it exists
            if current_entity:
                entities.append({'text': " ".join(current_entity), 'tag': current_tag_type})
            current_entity = [token]
            current_tag_type = tag.split('-')[1:] # e.g., ['Neg', 'Disease']
            current_tag_type = "-".join(current_tag_type)
        elif tag.startswith('I-') and current_entity:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append({'text': " ".join(current_entity), 'tag': current_tag_type})
                current_entity = []
                current_tag_type = None
                
    if current_entity:
        entities.append({'text': " ".join(current_entity), 'tag': current_tag_type})
        
    return entities