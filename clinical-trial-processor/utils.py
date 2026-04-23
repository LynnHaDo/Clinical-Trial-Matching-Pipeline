from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Make sure that sentences are in the same length
    """
    
    sentences = [item[0] for item in batch]
    tags = [item[1] for item in batch]
        
    # Pad sequences with 0s
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0) # 0 is the 'O' tag
        
    # Create the mask: 1 for real words, 0 for padding
    mask = (padded_sentences != 0).byte()
        
    return padded_sentences, padded_tags, mask