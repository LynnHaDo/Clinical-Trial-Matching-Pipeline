import torch 

def get_embedding(text, tokenizer, model):
    """Convert a medical term into a 768-d vector"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()