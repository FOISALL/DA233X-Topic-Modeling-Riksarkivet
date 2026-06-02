= Conclusions and Future Work <conc>

== Conclusions <conc:conc>

In conclusion, Using BERTopic appears to be the best course of action regarding the dataset at Riksarkivet. It provides the strongest coherence accompanied by a great diversity. Additionally it also runs the fasting given that the embeddings have been generated.

BERTopic can also be modified to ensure even noisy documents are given a topic at minimal penalty to quality of topics.

CTM falls behind BERTopic in terms of coherence but was shown to be the strongest model in terms of diversity instead.

LDA was shown to still produce competitive results on this dataset despite the struggles it was predicted to face. This showed that even though context based models performed the best, classical models should not be entirely disregarded.

Finally, this study showed that BERTopic consistently performs better when using embeddings generated without stopwords, even when they are just partially removed. This is highly beneficial as it allows the model to run faster and handle larger text within its context window.

== Limitations <conc:limitations>

The main limitation is that the evaluation is primarily based on intrinsic metrics. Topic coherence and topic diversity are useful for comparing models, but they do not fully capture whether the topics are meaningful to human users. A model may score well while still producing topics that are difficult to interpret, too broad, or not useful for information retrieval. The is also the added dimension that neural models sometimes scoring lower in the metrics because they might group topic based on wider context not measured by the metrics. In a larger study, a set of human evaluators could have been utilized to provide human input as contrast to the intrinsic metrics

Another limitation is that the results are based on a specific sample of the entire dataset. Generating the BERT embeddings took a significant amount of time so running it a large number of times or for the whole dataset proved problematic within the time limits of this study. The access to the GPUs was also limited which made longer runs difficult to complete. Different preprocessing choices, embedding models, hyperparameters, or document samples could change the ranking of the models. This is especially important for BERTopic when it came to handling noise. This study only tested one specific configuration of modifying the @hdbscan parameters, but it is possible that another configuration might produce very different results. 

The dataset itself is challenging because it contains OCR errors, historical spelling variation, and Swedish linguistic properties such as compounds and inflections. These factors make it difficult to know whether lower scores are caused by model limitations, preprocessing limitations, or the inherent difficulty of the data.

This study was also done without expert knowledge of the dataset. A better understanding of the content of the dataset could have guided the evaluation more around details such as an expected optimal number of topics and expected content of said topics. The stopword list generated could also have been more curated with more knowledge of the dataset to remove dataset specific stopwords. 

== Future Work <conc:fw>

This study compared multiple different models and how they performed on the specific dataset at Riksarkivet. This provided great insight into what worked well and which models struggled. There are however other relevant models that were not compared in this study. One such model is Top2Vec. This model has historically been compared alongside BERTopic as they both are able to capture context when creating topics. Previous work usually show that BERTopic creates higher quality topics, but due to the uniqueness of this dataset, it would still be valuable to compare alongside the other models in this study. A future study would therefore be encouraged to add Top2Vec to the models to be evaluated. 

There are several improvement that could be added to this study by having expert knowledge of the data, as described in @conc:limitations it would be interesting to see the improvements when using a more carefully fracted stopword list. Using Lemmatizers to preprocess the data could also be an interesting extension. It was disregarded in this study due to the complexity of applying it to both noisy and historical data. 

Expert knowledge would have also opened up the possibility to explore seeded topic models. As described in @bg:histdata, seeded models are a powerful option when working with historical data as it would allow the models to form topics around certain words, and also bind together concepts represented by different words across different times.

Observing the percentage of documents labeled as noise for BERTopic-Base in @table:bertbasenoise we see that when using the @d2 embeddings, the noise is notably lower for $K = 10$ and $K =50$. This is because one of the 3 runs used to calculate the average only classified below 10% of documents as noise. This seemed to randomly occur sometimes but has not been investigated further in this study. These runs appeared to be associated with lower overall metric scores but could still be interesting to investigate further in a future study.

== Reflections

My first thoughts about completing this project is that I am very satisfied that it was able to produce sensible results that are able to match up against previous work. Both the coherence and diversity scores were comparable and competitive with those of over studies.

This project also helped me value the importance of putting a lot of work into performing a pre-study on the material. Prior to this thesis, the topic model I had the most experience with was the ETM. The ETM later turned out to be the worst performing model in the study. Discovering to use BERTopic was rather straight forward, as it is one of the most popular and also recent models. the CTM however was not as visible and was a model a had not heard about before this thesis. CTM was only something I learned about when reading previous literature and comparative studies. Even though CTM did not perform as well as BERTopic it still contributed a lot to the study by displaying the highest diversity score among all the models.

Additionally, one of the central comparisons performed in the thesis was that of using @d1 or @d2 embeddings. This was an aspect which I had not even considered into rather late in the research process when I came across the study by murugaraj et. al. @murugaraj-etal-2025-mining while reading about topic models on historical data. Testing this turned out to be one of the more interesting parts of the study as it was nonintuitive that BERTopic would actually perform better when removing semantic structure from the documents.

Considering the fact that both CTM and BERTopic outperformed LDA and ETM, aswell as ETM and LDA struggling with the size of the dataset at Riksarkivet, it might have been more fruitful to to focus on evaluating more of the modern models. I mentioned Top2Vec in the discussion and how that might have been worth trying on the dataset as well. It also uses embeddings so it could also have utilized the BERT embeddings. It could also have been interesting to try different Language models for generating the embeddings, but using BERT is probably the most promising approach since we have the BK-BERT trained on Swedish by Kungliga Biblioteket.

I was very satisfied with the effectiveness of the stopword removal. Even though it was arguably not done very professionally. I described in the discussions method section, I only used a simple stopword list for modern Swedish, and then just asked generative AI to make a corresponding list but for old Swedish. I also added many words from the topics that were generated during trial runs. The list was not complete as the final topics still contain many words that should be considered stopwords or removed for other reason. The process of removing stopwords this way could probably go on for a very long time. This is why I am very satisfied with the fact that the stopword removal was still very helpful for the models.

It was a bit problematic for me to run the data on the full datasets. When I first tried to generate the embeddings it was estimated to take 48 hours for just the light embeddings. Thus, I opted to keep all my test to a 150k sample of the data. This worked fine to get my result and for comparing the models. When I at the end of the project tried again to get embeddings for the full dataset, it only took around 4-5 hours, a perfectly reasonable time. I do not know what changed to cause this difference, or if it had something to do with the GPU I was using.

Regardless, the ETM and LDA struggled with being run on the full dataset, while CTM and BERTopic where able to just generate topics on a sample and then extend them to the rest of the data. There are still ways to apply ETM and LDA to the full data but then it has to be done one part at a time and comes with a lot of work. It would have been valuable to explore but would probably have been outside the scope of this project

I have mentioned it earlier in the report but the evaluation metrics used in this study are not perfect. As described in the background they are sometimes not as accurate on contextualized models compared to other models. they also create a gap between recorded values and actual human interpretation of the results. It would have been very interesting to attempt to apply human evaluation on the results. Though this would not have been impossible, it did not feel feasible for me to organize in the timeframe of this thesis. Most importantly however, human evaluation is not perfect as it introduces a lot of uncertainties in how the results should be interpreted and is overall more difficult to handle than the more straight forward intrinsic metrics.


