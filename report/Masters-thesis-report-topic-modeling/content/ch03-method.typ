= Method <mth>

This chapter aims to in detail describe the method and steps taken in the study to compare the different topic models in the dataset of Riksarkivet. and how these results are evaluated. It will describe how the results are obtained and how they will be evaluated. @mth:rp will go through all the steps of the method, describing the overall process. @mth:data goes into further detail into the data, its properties and how it is handled, as that is a significant part of the work in this study. @mth:tm goes into detail on the topic models, describing how they were selected and any setup needed in order to properly run them, such as fine tuning. @mth:exp covers the software tools and frameworks used, as well as an overview of the hyperparameters used. @mth:val covers a self scrutiny of the robustness of the method. It covers the evaluation metrics but also the process itself and reasons about the validity and reliability of the method.


== Research Process <mth:rp>

This thesis will conduct a comparative study between four different topic models, LDA, ETM, CTM and BERTopic. The goal is to answer the research question _What topic modeling techniques produce the best results on the dataset at Riksarkivet?_ as described in @intr:prbl. This study will use predefined evaluation frameworks to run and evaluate the models. This means that a significant part of the technical work will focus on how to best preprocess and apply the data to the models. The research processes will be described in detail to allow for reproducibility. It will follow the following overarching steps

#enum(indent: 2em, spacing: 1em,
[
  *Literature review*:\
  A literature study will be conducted to research the relevant topics of the domains, it will focus on answering the questions:
    - What are the most interesting models and what are their strengths and weaknesses in regards to the dataset?
    - How have other studies compared topic models before?
    - how have studies in the domain of topic modeling handled similar datasets in the past?
    - What evaluation metrics are common in the domain and how reliable are they
],
[
  *Preprocessing of data*:\
  The data must be preprocessed in order to accommodate all the models. This will require the following steps:
  - Parse the data from the OCR transcribed xml files.
  - Lightly preprocess the corpus to produce a full text version for the neural models. This includes repairing words split by line breaks and lowercasing the text
  - Further preprocess the text by removing stop-words and words rendered incomprehensible by OCR noise.
  - store the data as both a version containing the full text for the BERT based models CTM and BERTopic, and a @bow representation version for ETM and LDA.
 
],
[
  *Fine tuning of models*:\
  In order to ensure the BERT based models can follow the semantic context of the dataset, the underlying BERT model must be fine tuned to the specific dataset
],
[
  *Framework setup*:\
  Set up the evaluation framework and the pipeline to use it. This includes:
  - Set up the OCTIS framework for use
  - Set up BERTopic from its original github for use
  - Applying the preprocessed dataset to framework.
  - Applying the finetuned SBERT models to CTM and BERTopic
],
[
  *Running models*:\
  The models will then be run; LDA, ETM and CTM will be run through the OCTIS framwork, while BERTopic will be ran on its own using the Gensim library. The models will be run for a different number of topics  $K in {10, 20, 30, 40, 50, 60}$ and their best performance will be compared.
  


  LDA and ETM will be run on the heavily preprocessed @bow dataset, while CTM and BERTopic will be run on the less preproccessed version that retains the semantic context. Following the method of @murugaraj-etal-2025-mining, both CTM and BERTopic will also be run on the first dataset as even neural models have been shown to perform better on heavily preprocessed data.

],
[
  *Evaluation*:\
    The results will be represented by a coherence and diversity score. The scores will be calculated based on the 10 most prominent words in the topic. If a topic does not contain even 10 prominent words, this topic will be disregarded for evaluation.

    In the case of BERTopic, Some documents will be assigned the "junk topic" instead of being assigned a proper topic. Therefore A separate analysis will also be conducted on BERTopic to determine what portion of the documents are actually assigned a proper topic.
],
[
  *Comparison of evaluation metrics*:\
  The evaluation metrics will then be compared against each other to conclude which models performed the best compared to each other and also for what number of topics.
],
[
  *analysis of results*:\
  Finally the results will be viewed in a wider context were the models ease of use, data preprocessing requirements and evaluation metrics will be viewed together to draw conclusions on which models are best suited for the dataset at Riksarkivet
]

)

#figure(
  image("overall-pipelinev1.drawio.png"), caption: [Overview of the entire project pipeline]
)


== Dataset <mth:data>

This section covers everything related to the dataset. section @mth:dc covers how the data was acquired. section @mth:dp covers the special properties of the dataset that the methodology will need to account for. @mth:ppr covers the various steps needed in order to preprocess the data.

=== Data collection <mth:dc>

The data used in this study is by Riksarkivet. It is part of their public records and thus, does not contain any sensitive information.

=== Data properties <mth:dp>

This section describes the specific properties of the data that will be required to take into consideration. @mt:ocr describes how the @ocr has affected the data and @mt:hv describes how the unique timespan of the documents effect the dataset.

==== OCR transcription <mt:ocr>

The data files contain text transcribed from handwritten documents using @ocr. It contains most of the common errors caused by @ocr described in @bg:ocr, such as misspelled words or words that have been separated or mixed together. Many words are also split by line breaks. The transcribed text has an estimated @cer of 0.03 to 0.1. 

==== Historical variations <mt:hv>

Riksarkivet stores documents as old as a thousand years. The documents that will be considered for this dataset spans roughly from 1600s to 1900s. This means that the Swedish used in the documents is different from the modern Swedish. Since these documents span several hundred years they will also have language variation compared to each other. Additionally they will also have many local variations depending on where in Sweden they originated from, since the national Swedish wasn't standardized before the 1800s.

=== Data preprocessing <mth:ppr>

Before the data can be used by the topic models, there are several preprocessing steps that need to be taken. This section will make an effort to explain the preprocessing steps. As noted by previous studies @hall_mernitz_rensch_2026, the preprocessing steps and hyper parameters are too often omitted from studies, making comparisons and reproductions difficult.

The data is stored as xml files containing the transcriptions as well as related metadata. These files must be parsed into readable documents. The raw data will be parsed into a single .jsonl, storing one document per row. The documents are stored with their additional metadata as 

```
{"doc_id": "id.xml", "source_image": "id.jpg", "processed_at": "date", "raw_text": "text", "word_count": 250}.

```

The BERT based models require the full texts in order to extract semantic meaning, while LDA and ETM just need the @bow version with the word frequencies. Because of this, the documents have to be stored twice. 
#figure(
image("preprocessv1.png")
,caption: [Overview of preprocessing pipeline]
)


==== Dataset 1 <mth:d1>

The BERT based models will use a lightly preprocessed dataset. This dataset will feature the following preprocessing steps:

#enum(indent: 2em, spacing: 1em,[
  *Repairing words hyphenated by line breaks:* Many words have been split up due to line breaks. An example of such a case is  ```"närmare mid¬\ndagen."```, where the word "middagen" has been split into "mid" and "dagen" split up by "```¬\n```". This word is stitched together using the regex : ```text = re.sub(r'[¬-]\s*\n\s*', '', text)```, replacing ```¬\n``` with the empty string. Isolated line breaks are also removed.
],[
  *Removing random noise:* Sometimes random symbols appear in the OCR, therefore, every single characters surrounded by white space is removed with the regex: ```re.sub(r'\s+[^\w\s.,!?]\s+', ' ', text)```. This will remove some single letter words like "ö" but has been deemed to ultimately improve the quality.
],[
  *White space normalization:* All large sections of consecutive whitespace is replaced with a single white space to avoid gaps in the text.
],
[
  *Lowercasing:* All capital letters are changed to lowercase for consistency. Other sentence structures such as punctuations and space are kept to retain the semantic structure of the text.
])

This yields us our lightly preprocessed dataset that retains the semantic structure of the text while having a reduced amount of noise compared to the raw dataset. This dataset will be refered to as @d1
==== Dataset 2 <mth:d2>
LDA and ETM, which require a more heavily preprocessed dataset will take the lightly processed dataset and further process it. This will be done by the following steps

#enum(indent: 2em, spacing: 1em,[*Noise removal:* This step will remove additional OCR noise that wasn't caught in the lightly pre processed dataset. This step focuses on removing the most rare words that have likely been created by misspellings or mistranscriptions. This will be done by deleting all words that only appear in a maximum of 15 documents.

],[
*Stopword removal:* Removing stopwords will be done in two steps, first based on frequency, and then based on a manually compiled blacklist removing common know stopwords.

- Since there is no comprehensive list of all historical Swedish stop words, they will be removed based on frequency. This has been done by other studies @hall_mernitz_rensch_2026 but with a much higher cutoff at only 50%. Since this study handles a dataset containing documents from several hundred years, the language will have changed to a much higher degree. Because of this larger timeframe, a lower cutoff of around 20% percent will be used. However this number will be treated as hyperparameter $alpha$. This method allows for removal of domainspecific stopwords, overrepresented in this specific dataset.

- To ensure a large amount of captured stopwords are removed a list of stopwords will also be compiled. Since no full list exists, the list used in this study will be imperfect. First a precomposed word list with modern Swedish stopwords will be used @Dahlgren_Svensk_Text_2018. This list will then be prompted to an AI agent in order to expand it with historical stop words. Lastly, after running the models, additional stop words will be added to the list based on manual inspection of the topics.
])

Unlike other studies, the data will not be run though a lemmatizer even if it is recommended for Swedish corpora. This is because lemmatizers greatly struggles with Historic data and would likely cause more harm to the dataset than it would actually help.

This is our heavily preprocessed dataset that can be used as a @bow representation of the documents and will be referred to as @d2.

== Topic Models <mth:tm>

=== Model Selection <mth:ms>

Based on previous research, neural models appear versatile and able to handle large datasets. For this reason both CTM and BERTopic have been chosen for evaluation. However despite this, LDA has remained a topic model used in many studies, and under the right circumstances still performing on par with or better than @ntm:pl. For this reason it has been deemed important to still include it in the study as a baseline. Though most recent research has been focused on @ntm:pl, they have yet to be fully adopted by researchers in other fields. For this reason it is scientifically interesting to compare the different model paradigms to each other. ETM is a neural model that still relies heavily on the @bow representation, therefore it will also be evaluated as sort of a middle ground between the two paradigms.

=== Model Fine-Tuning <mth:ft>

Since the LDA exclusively uses the @bow lacking semantic understanding and ETM utilizes static embeddings, they can be applied to the preprocessed dataset directly. CTM and BERTopic however use SBERT to grasp the semantic meaning from the text. This means that their understanding of the text is only as good as the underlying SBERT model. Though CTM has the capabilities for so called zero shot learning without fine-tuning, this relies on the fact that the pre-train of SBERT was done on the same domain, which in this case it is not @bianchi_terragni_hovy_nozza_fersini_2021. 

To resolve this, a pre-trained BERT model for Swedish data is utilized. The model used for this will be KB-BERT which has been trained on billions of words sourced from Swedish newspapers, government reports, social media, wikipedia and more. This model will then be fine-tuned in an unsupervised manner by training it on the raw text of the documents from the dataset. The training is done using @tsdae. Since we have no shortage of data, a sample of 200k documents will be used for the fine tune, but as described in @wang-etal-2021-tsdae-using, less than 100k is usually enough for proper domain adaptation. Increasing the amount of data leads to diminishing returns.

First the sample is fetched from the lightly preprocessed dataset. Then the KB-BERT model is loaded. A mean pooling layer is then attached to the BERT-model in order to convert the token embeddings into a single fixed size document embedding for each document. The data is then split into batches of 8 documents each. The batches are then masked and fed into the encoder-decoder network. Once the training is done, the final Encoder is the finished SBERT model which will be used.

CTM and BERTopic require document level embeddings instead of word level embeddings. With the BERT model fine tuned and adapted to an SBERT model, it is now ready to be applied to CTM and BERTopic. This is done using a library that provides a pooling layer that can convert all the vectors of a document into one single vector using mean pooling. This gives every document one single vector that represents them.

=== BERTopic Variants

BERTopic's @hdbscan clustering step is notorious for sorting documents into a "junk" topic if they are deemed unfit for any of the "real" topics. Due to the noisy nature of the documents in the dataset, this may end up affecting a significant portion of the documents. To provide Riksarkivet with alternatives, two additional variations of BERTopic that mitigate this problem will be evaluated. 

- On top of the base BERTopic model, a version that modifies the parameters of @hdbscan will be used. Specifically, the "min_samples" variables was decreased from the standard 10 to 5. This allows points to associate with clusters despite having a less neighbors.

- Alongside the afformentioned version, we will also use a varaition that keeps the base @hdbscan parameters but forcibly reassigns all the documents to their best fitting "real" topic.

=== ETM Variants

One of the strengths of the ETM is that its ability to handle larger vocavularies without loss of performance as described by dieng et al. @dieng_ruiz_blei_2020. Theoretically this propertly should be benificial to handle the dataset at Riksarkivet, as the abundance of words from different timeframes implies a larger vocabulary. To investigate if this property holds, two versions of the ETM will be evaluated. One version will use a vocabulary of 2000, and a second version will use a much larger vocabulary of 6000.

== Experimental Setup <mth:exp>

=== Software and Frameworks
To ensure standardized comparisons, LDA, ETM, and CTM will be implemented and trained using the @octis @terragni-etal-2021-octis framework, while BERTopic will be run via its official library as it is not available in @octis. The @octis environment was set up via a cuda command prompt.

@octis uses the Gensim Python library in order to access the topic models.

To run the models quicker, KTH GPU resources were used.

=== OCTIS 

@octis is a framework made for training, analyzing and comparing topic models. It provides a stable pipeline to ensure reliable and reproducible results. It does currently not support BERTopic but will be used to trained the remaining models: LDA, ETM and CTM.

The framework uses the Gensim library in order to access the models and their fucntionalities

Even though it cannot train and run the BERTopic model it can still evaluate its results. Thus @octis main role will be to  evaluate and compare the models.  



=== Hyper parameters
There are a few hyper parameters that will be considered in this study.

The number of topics $K$ that the models will generate. The models will be tasked with generating a span of 5 different number of topics, eg $K in {20, 30, 40, 50, 60}$ to see which ones yield the best results. The optimal number of topics may vary from model to model. 

The cutoff percentage $alpha$ for what is to be considered stop words in the dataset. This parameter will be tweaked slightly to investigate its impact on the result.





=== Evaluation <mth:ev>

The output of the models will be evaluated using two intrinsic evaluation metrics:

#list(indent: 2em,spacing:2em,[
      *Topic Coherence @npmi*: Measures the semantic interpretability of the topics by calculating the co-document frequency of the top 10 words in a topic. The metrics scales from 0 to 1, where a score closer to 1 means a better coherence.
  
],
[
      *Topic Diversity*: Measures the fraction of unique words across all generated topics to penalize models that produce redundant or collapsing topics @dieng_ruiz_blei_2020. A score of 0 indicates the worst possible diversity, while a score of 1 represents the best possible result with perfectly distinct topics.
])

As described in section @bg:mtr these metrics are not perfect and sometimes misleading when comparing neural models against non-neural models. The final conclusions will therefore combine the aforementioned metrics together with the runtime for each model and a general difficulty of use. Difficulty of use could constitute aspects such as data preprocessing requirements and fine tuning needed for the model to perform.

== Validity and Reliability of Method <mth:val>

This study attempts to maximize validity by mimicking the methodologies of previous studies. None of the topic model architectures will be reimplemented or modified. Instead predefined models will be used. The evaluation framework @octis will be used to run the models. This ensures the reproducibility of the study. Unfortunately there exists imperfections in the study due to the nature of the field.

The evaluation metrics coherence and diversity are imperfect and have been observed to not always correspond to better results for human interpretation. Unfortunately these are state of the art most common metrics applied to topic models. The results should therefore be viewed as a guideline and not conclusive evidence.

The study attempts to preprocess the data in a way that will benefit the models in the best way possible but it is impossible to know if another kind of preprocessing would yield better results. As shown by @murugaraj-etal-2025-mining, applying a neural model to a heavily preproccesed dataset without semantic context unexpectedly yielded better results. The nature of the data also makes it difficult to properly preprocess and it is possible that other methods than the ones applied in this paper may have been better suited for handling historical Swedish data.

To allow for fair comparisons all models will use the same @bow:pl representations. It is however possible that BERTopic benefits more from a different kind of preprocessing that what works best for LDA.

The fine tuning of SBERT models is also inherently probabilistic. Even if a reproduction is done by fine tuning on the same dataset, there is no guarantee that the fine tuning will be exactly the same.

Despite this the paper follows the best proven methods in the fields and the quality of the results should achieve a similar standard as previous studies.









