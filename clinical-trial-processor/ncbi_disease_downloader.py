from datasets import load_dataset
from constants import NCBI_DATASET_NAME, LOCAL_NCBI_DATASET_DISK_PATH
import os

print("Downloading and loading NCBI-Disease dataset...")
dataset = load_dataset(NCBI_DATASET_NAME, trust_remote_code=True)

export_path = os.path.abspath(LOCAL_NCBI_DATASET_DISK_PATH)
dataset.save_to_disk(export_path)

print(f"Dataset loaded successfully and saved for offline use at: {export_path}!")
print("Splits:", dataset.keys())

# Print the first training example to verify
print("\nFirst training record:")
print(dataset['train'][0])
