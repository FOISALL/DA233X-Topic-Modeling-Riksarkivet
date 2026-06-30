import pandas as pd

# Load the full, lightly preprocessed dataset
print("Loading full dataset...")
df_full = pd.read_parquet("riksarkivet_neural_cleaned.parquet")

# Create the sample (e.g., 150,000 documents)
print("Sampling 150,000 documents for the Evaluation Set...")
df_sample = df_full.dropna(subset=['clean_text']).sample(n=150000, random_state=22)

# Save the Sample
sample_file = "riksarkivet_final_neural_sample_150k.parquet"
df_sample = df_sample.reset_index(drop=True)
df_sample.to_parquet(sample_file, index=False)

print(f"Success! Sample saved to {sample_file}")
print("Use ONLY this file for all downstream tasks.")