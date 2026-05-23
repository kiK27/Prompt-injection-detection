from datasets import load_dataset
import pandas as pd

print("Downloading dataset...")

dataset = load_dataset(
    "cyberec/Prompt-injection-dataset",
    "core"
)

train_df = pd.DataFrame(dataset["train"])

train_df.to_csv(
    "models/prompt_injection_dataset.csv",
    index=False
)

print("Dataset saved successfully!")