import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models, datasets, losses
from torch.utils.data import DataLoader

import nltk

# NLTK tokenizers!
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the FULL dataset
print("Loading the full dataset...")
df = pd.read_parquet("riksarkivet_neural_cleaned.parquet")

# Remove the .sample() function to use all ~2 million documents
train_sentences = df['clean_text'].dropna().sample(n=200000, random_state=42).tolist()
print(f"Loaded ALL {len(train_sentences)} documents for training.")

# Load the KBLab BERT model and convert to SBERT
print("Loading KBLab base model...")
word_embedding_model = models.Transformer('KBLab/bert-base-swedish-cased', max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

# Prepare the TSDAE DataLoader
print("Preparing the TSDAE DataLoader, may take a minute...")
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

# chat gpt suggestion:
# Batch size of 8 is safe for 20GB vRAM. If you get a CUDA Out of Memory error, drop this to 4.
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define the TSDAE Loss function
train_loss = losses.DenoisingAutoEncoderLoss(
    model, 
    decoder_name_or_path='KBLab/bert-base-swedish-cased', 
    tie_encoder_decoder=False
)

# Train the Model!
print("Starting TSDAE Fine-Tuning on 2 MILLION documents.")
print("A progress bar will appear below showing your ETA...")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True, 
    checkpoint_path='sbert_historical_checkpoints-200k', 
    checkpoint_save_steps=10000, # Saves a backup every 10,000 batches
    checkpoint_save_total_limit=2 
)

# Save the final model
output_path = "riksarkivet-sbert-historical-swedish-200k"
print(f"\nTraining complete! Saving final model to {output_path}...")
model.save(output_path)

print("Success! Your custom SBERT model is ready.")