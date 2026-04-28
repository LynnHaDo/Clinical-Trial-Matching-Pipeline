import torch
import re
from torch.nn.utils.rnn import pad_sequence
from constants import AACT_DB_NULL_VALUES, COMMON_DRUG_SUFFIXES, DATASET_VOCAB_KEYS

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

def prepare_sequence(nlp_model, seq, to_ix):
    """
    Converts a text string into a tensor of vocabulary IDs.
    """
    if not seq:
        return torch.tensor([], dtype=torch.long)
    
    clean_text = re.sub(r'([.,:;!?()])', r' \1 ', str(seq)) # Add spaces around punctuation
    doc = nlp_model(clean_text)
    tokens = [token.text for token in doc]
    idxs = [to_ix.get(w.lower(), to_ix[DATASET_VOCAB_KEYS.UNKNOWN.value]) for w in tokens]
    return torch.tensor(idxs, dtype=torch.long), tokens

def extract_entities(tokens, tags):
    """
    Parses tags back into actual phrases
    """
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
    
    for entity in entities:
        entity['text'] = re.sub(r'[.,:;!?()]', '', entity['text'].strip())
        
    return entities

def split_criteria(text):
    """
    Splits ClinicalTrials.gov `criteria` entry into inclusion and exclusion strings
    
    Returns: a tuple of (inclusion text, exclusion text)
    """
    if not text:
        return "", ""
    
    # Look for exclusion criteria and everything after it
    match = re.search(r'(?i)exclusion criteria[:\-]?\s*(.*)', text, re.DOTALL)
    
    if match:
        exc_text = match.group(1)
        inc_text = text[:match.start()] # everything before is inclusion criteria
        inc_text = re.sub(r'(?i)inclusion criteria[:\-]?\s*', '', inc_text)
        return inc_text, exc_text
    else:
        return text, ""

def clean_lines(textblock):
    """Strips leading special characters (bullets) from block of text"""
    lines = textblock.split('\n')
    return [re.sub(r'^[\*\-\+]\s*', '', line.strip()) for line in lines if line]

def normalize_age(age_str):
    """
    Extract age (XX Months) from ClinicalTrials.gov `minimum_age`/`maximum_age`
    
    Based on documentation from AACT Database:
    Definition: The numerical value, if any, for the minimum/maximum age a potential participant must meet to be eligible for the clinical study.

    Unit of Time * Select one.
    
        Years
        Months
        Weeks
        Days
        Hours
        Minutes
        N/A (No limit)
    
    """
    if not age_str:
        return None
        
    # Clean string and handle known null values
    age_str = str(age_str).strip().lower()
    if age_str in AACT_DB_NULL_VALUES:
        return None
        
    # Regex extracts the number (including decimals) and the unit
    match = re.search(r'([\d\.]+)\s*(years?|months?|weeks?|days?|hours?|minutes?)', age_str)
    
    if not match:
        return None
        
    value = float(match.group(1))
    unit = match.group(2)
    
    # Standardize to Months
    if unit.startswith('year'):
        return value * 12
    elif unit.startswith('month'):
        return value
    elif unit.startswith('week'):
        return value / 4.345 
    elif unit.startswith('day'):
        return value / 30.437
    elif unit.startswith('hour'):
        return value / (30.437 * 24)
    elif unit.startswith('minute'):
        return value / (30.437 * 24 * 60)
        
    return None

def clean_dataset_name(datasetName):
    return re.sub(r'/', '_', datasetName)

def is_valid_medical_term(nlp_model, text):
    """
    Returns true if the text contains a recognized medical entity
    """
    doc = nlp_model(text)
    return len(doc.ents) > 0

def process_entities_from_text_chunks(text_chunks, nlp_model, word_to_ix, model, ix_to_tag, trial_graph, inclusion=True):
    for text_chunk in text_chunks:
        if len(text_chunk.split()) < 2: continue
            
        inputs, tokens = prepare_sequence(nlp_model, text_chunk, word_to_ix)
        mask = torch.ones(1, len(inputs), dtype=torch.bool)
        with torch.no_grad():
            predicted_tag_ids = model.decode(inputs.unsqueeze(0), mask)[0]
            
        predicted_tags = [ix_to_tag[tag_id] for tag_id in predicted_tag_ids]
        extracted_entities = extract_entities(tokens, predicted_tags)
        
        for entity in extracted_entities:
            if is_valid_medical_term(nlp_model, entity['text']):
                tag = entity['tag']
                text_lower = entity['text'].lower()
                
                if tag == 'Disease' and text_lower.endswith(COMMON_DRUG_SUFFIXES):
                    tag = 'Chemical'
                elif tag == 'Neg-Disease' and text_lower.endswith(COMMON_DRUG_SUFFIXES):
                    tag = 'Neg-Chemical'
                
                if inclusion:
                    if tag == 'Disease':
                        trial_graph["edges"].append({"type": "REQUIRES_CONDITION", "target": entity['text']})
                    elif tag == 'Chemical':
                        trial_graph["edges"].append({"type": "REQUIRES_CHEMICAL", "target": entity['text']})
                    if tag == 'Neg-Disease':
                        trial_graph["edges"].append({"type": "EXCLUDES_CONDITION", "target": entity['text']})
                    elif tag == 'Neg-Chemical':
                        trial_graph["edges"].append({"type": "EXCLUDES_CHEMICAL", "target": entity['text']})
                else:
                    if entity['tag'] == 'Disease':
                        trial_graph["edges"].append({"type": "EXCLUDES_CONDITION", "target": entity['text']}) # any disease found here are excluded
                    elif entity['tag'] == 'Chemical':
                        trial_graph["edges"].append({"type": "EXCLUDES_CHEMICAL", "target": entity['text']})