from datasets import load_dataset
import os

print("Downloading and loading NCBI-Disease dataset...")
cache_folder = os.path.abspath('./datasets')
dataset = load_dataset("ncbi_disease", trust_remote_code=True, cache_dir=cache_folder)

print("Dataset loaded successfully!")
print("Splits:", dataset.keys())

# Print the first training example to verify
print("\nFirst training record:")
print(dataset['train'][0])
