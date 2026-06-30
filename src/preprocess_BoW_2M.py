import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from tqdm import tqdm
import logging
import string
import unicodedata
import re 

#Setup Logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO
)

# Expanded Historical Stopwords
f = open("allstopwords.txt", "r")
historical_stopwords = f.readlines()
historical_stopwords = [word.replace('\n','') for word in historical_stopwords]
class MyCorpus:
    def __init__(self, path):
        self.path = path
        self.length = len(pd.read_parquet(path, columns=[])) 
        self.remove_punct = str.maketrans('', '', string.punctuation)
        self.remove_nums = re.compile(r'\d+')

    def __iter__(self):
        # read the parquet in one go here for speed, but process document by document
        df = pd.read_parquet(self.path, columns=['clean_text'])
        
        for text in df['clean_text']:
            if text:
                # FIX UNICODE (NFC normalization for Swedish åäö)
                clean_str = unicodedata.normalize('NFC', str(text).lower())
                
                # DELETE PUNCTUATION
                clean_str = clean_str.translate(self.remove_punct)

                

                # remove digits
                clean_str = self.remove_nums.sub('', clean_str)
                
                # TOKENIZE & FILTER (Length > 2 and NOT in stopwords)
                tokens = [
                    word for word in clean_str.split() 
                    if len(word) > 2 and word not in historical_stopwords
                ]
                yield tokens

    def __len__(self):
        return self.length

# --- EXECUTION ---

input_file = "riksarkivet_neural_cleaned.parquet"
streamed_corpus = MyCorpus(input_file)

# Build Dictionary
print("Building Dictionary...")
dictionary = Dictionary(tqdm(streamed_corpus, total=len(streamed_corpus), desc="Dictionary"))

# Filter extremes (using your alpha/tau logic)
print("Filtering extremes (no_below=15, no_above=0.20)...")
dictionary.filter_extremes(no_below=15, no_above=0.20)
dictionary.compactify()

# Save Dictionary
dictionary.save("riksarkivet_2M_bow_dictionary_20.dict")

# Serialize BoW Corpus
print("Serializing BoW Corpus...")
MmCorpus.serialize(
    "riksarkivet_2M_bow_corpus_20.mm", 
    (dictionary.doc2bow(doc) for doc in tqdm(streamed_corpus, total=len(streamed_corpus), desc="Saving BoW")),
    progress_cnt=10000
)

print("\nSuccess!")