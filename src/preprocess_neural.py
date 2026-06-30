import json
import re
import pandas as pd
from tqdm import tqdm

def clean_for_neural_models(text):
    """
    Cleans raw OCR text for Transformer models (BERTopic/CTM).
    Retains grammar, stopwords, and sentence boundaries (. ! ?).

    This is the file that creates the D1 dataset, 
    used for finetuning and sometimes generating the embeddings of CTM and BERTopic

    outputs the file "riksarkivet_neural_cleaned.parquet"
    """
    if not isinstance(text, str):
        return ""
    
    # OCR De-hyphenation (fixes 'up¬\nkallande' -> 'upkallande')
    text = re.sub(r'[¬-]\s*\n\s*', '', text)

    # Remove any lingering soft hyphens that weren't part of a line-break
    # This targets the character ¬ specifically, even if it's attached to a word.
    text = text.replace('¬', '')
    
    # Remove remaining line breaks
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Clean up isolated OCR garbage (keeps punctuation for BERT)
    text = re.sub(r'\s+[^\w\s.,!?]\s+', ' ', text)
    
    # Whitespace Normalization
    text = re.sub(r'\s+', ' ', text)
    
    # Lowercase standardization
    text = text.lower().strip()
    
    return text

def process_and_save(input_file, output_file):
    processed_documents =[]
    
    # Count total lines so the progress bar can calculate an exact ETA
    print(f"Scanning {input_file} to count total documents...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f)
    print(f"Found {total_lines} documents. Starting text cleaning...")
    
    # Process the file line-by-line with a progress bar
    with open(input_file, 'r', encoding='utf-8') as file:
        # tqdm wraps around the file iterator to give you the progress bar
        for line in tqdm(file, total=total_lines, desc="Cleaning OCR Text"):
            if not line.strip():
                continue
                
            # Parse the JSON line
            doc = json.loads(line)
            
            # Extract relevant fields
            doc_id = doc.get("doc_id", "unknown")
            raw_text = doc.get("raw_text", "")
            word_count = doc.get("word_count", 0)
            
            # Clean the text
            clean_text = clean_for_neural_models(raw_text)
            
            # Append to our list
            processed_documents.append({
                "doc_id": doc_id,
                "clean_text": clean_text,
                "original_word_count": word_count
            })
            
    # Step 3: Convert to Pandas and Save
    print("\nConverting to Pandas DataFrame...")
    df = pd.DataFrame(processed_documents)
    
    print(f"Saving heavily compressed Parquet file to {output_file}...")
    # index=False prevents pandas from saving row numbers, saving even more space
    df.to_parquet(output_file, index=False) 
    
    print("Done! Your neural dataset is ready.")

# --- EXECUTE THE SCRIPT ---
if __name__ == "__main__":
    # CHANGE THIS to the actual name/path of your raw data file
    INPUT_JSONL = "historical_corpus_full_raw.jsonl" 
    
    # This is the file it will create. You will load this in your future notebooks!
    OUTPUT_PARQUET = "riksarkivet_neural_cleaned.parquet"
    
    process_and_save(INPUT_JSONL, OUTPUT_PARQUET)