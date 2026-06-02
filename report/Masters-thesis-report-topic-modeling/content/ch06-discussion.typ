= Discussion <disc>

This chapter discusses the results presented in @res. The discussion first returns to the research question and hypothesis stated in @intr:prbl, before interpreting the main findings from the model comparison. The chapter then discusses the influence of preprocessing, the role of noise handling in BERTopic, the trade-off between coherence, diversity, and runtime, and finally the limitations of the evaluation.

== Research Question and Hypothesis <disc:rq>

The research question presented in @intr:prbl of this thesis was: _What topic modeling techniques produce the best results on the dataset at Riksarkivet?_ Based on the results in @res:sum, BERTopic produced the strongest overall results. The best-performing configuration was the Base BERTopic model when using the @d2 embeddings for $K = 30$.

This result partially supports the hypothesis that BERT-based models would outperform the alternatives. Both BERTopic and CTM performed competitively, with BERTopic achieving the highest coherence and CTM achieving the highest diversity. This reinforces the idea that contextual models are better suited for the noisy and historically varied documents used in this study. 

The results also show that the advantage of neural or contextual models is not automatic. BERTopic performed well, But even though CTM achieved the next highest score, it was still a notable decrease compared to BERTopic and was more similar to the LDA. This suggest that even though BERT-based models are definitely a good fit for the dataset, it is specifically BERTopic that stood out. other methods should still be considered as an option. 

CTM also achieved the high diversity but required substantially longer runtime. ETM, despite being a neural topic model, performed much worse than LDA in terms of coherence. 

The drawback of BERTopic is that almost half of the documents are are classified as noise instead of being assigned to a topic. This ensures a higher level of relevancy of the documents in the topics. If Riksarkivet however finds it important that all documents are classified in topics, using the reassignment strategy for BERTopic actually shows the second best overall results. Though it takes a hit in diversity, it only scores slightly lower than the base model in coherence. 

Based on these results, BERTopic is the most suited model for the dataset at Riksarkivet. Not only because of its high coherence, but because it manages to balance it with both a high diversity and the lowest runtime. It can also be easily modified to ensure no documents are classified as noise without a major score penalty.

== Coherence Scores <disc:doherence>

BERTopic scored the highest in coherence, this is consistent with the findings of @murugaraj-etal-2025-mining where BERTopic also scored visibly higher compared to LDA. The results of this study still fall short in score compared to the best results of aorund 0.21 @tc achieved by @murugaraj-etal-2025-mining but is still towards the higher end, beating out every model except their very best. The coherence of the LDA also exceeded the values presented by @murugaraj-etal-2025-mining, indicating that the dataset of Riksarkivet may be more well behaved than other noisy and historical datasets where topic modeling has been proven to work.

CTM Managed to score higher than all models except BERTopic. This is also in line with previous studies. @abdelrazek_eid_gawish_medhat_hassan_2023 presented results showing the CTM scoring higher than both ETM and LDA, with ETM falling behind the LDA. This Study showed similar result, but the CTM and LDA were much more equal here compared to other studies.

The paper introducing BERTopic @Maarten_G_2022 also compared it to CTM. Their results showed, similar to those of this study, that BERTopic achieves a higher coherence than CTM. In their case however the difference was rather extreme, with BERTopic achieving almost twice the score of CTM. In our case the difference was meaningful but CTM still achieved very respectable scores in comparison.

The overall scores are not only comparable to the scores presented in @murugaraj-etal-2025-mining who also handles historic data. But is also competitive with the scores from @abdelrazek_eid_gawish_medhat_hassan_2023 as well as the papers that introduced the models. This shows that at least according to the evaluation metrics, we are able to achieve results on the problematic dataset at Riksarkivet that is comparable to those achieved by other studies on more well behaved datasets.

The LDA scores surprisingly well, showing results competitive with the CTM. This is surprising as it was expected to struggle with the noisy nature of the dataset aswell as the large vocabulary, which it mostly has to disregard as it only considers a small portion of it.

The LDAs succes might be explained by it achieving large benefits from from only considering a small subset of the vocabulary, thus disregarding a large amount of noise. This should not be the full explanation as it scores much better than the ETM which does the same thing. Interestingly though, the ETM scores notably worse when attempting to utilize a larger vocabulary.

== Diversity Score <disc:diversity>

The diversity score, measuring how distinct the topics were from each other, also showed similar results to previous studies.

BERTopic, which scored the highest in coherence, does fall behind CTM in diversity. This aligns with the results presented in @Maarten_G_2022 where The newly introduced BERTopic exceeded the CTM in coherence and while still achieving a competitive diversity, does not manage to surpass the CTM. Even in the study of @abdelrazek_eid_gawish_medhat_hassan_2023, CTM scores the highest in diversity compared to both the ETM and LDA. ETM also performed particularly bad in diversity in their study which matches our results. Particularly, the ETM had the most visible K dependent diversity out of all the models in this study. 

Though the ETM has been shown to perform bad in @abdelrazek_eid_gawish_medhat_hassan_2023, the poor results in this study is still somewhat unexpected as the dataset description does seem to cater towards its strengths of being able to handle larger vocabularies and an abundance of rare words described in @dieng_ruiz_blei_2020. 

All models showed a drop off in diversity as the number of topics increased, except for the LDA. This made intuitive sense as it becomes harder to keep topics distinct and the room for redundancy increases as more topics are created. ETM displayed the most clear example of this relationship as seen in @fig:etmdiv where the diversity quickly drops.



== Noise Classifications in BERTopic <disc:bertopic-noise>

A central finding is that BERTopic's treatment of outliers strongly affected its usefulness. The base BERTopic model classified a large proportion of documents as noise, in many cases close to or above half of the dataset. This is problematic because the purpose of topic modeling in this thesis is not only to create coherent topics, but also to organize documents in a way that could support future information retrieval.

The modified @hdbscan parameters reduced the number of noise documents from around 50% to 35-40%. This is a meaningful decrease but still a very large amount of noise. The downside is that it came at the cost of a rather large decrease in coherence. This indicates that simply forcing the clustering algorithm to accept more documents into clusters can reduce topic quality.

In contrast, the reassignment strategy managed to remove the noise category while still preserving strong coherence. The results were however more varied across runs and different number of Topics. There was still a noticeable loss in diversity but altogether this makes reassignment the more promising BERTopic variation for this dataset.

Whether the different strategies for noise reduction should be used at all is up to the end user. If full coverage is of great importance then the reassignment strategy can achieve that without interfering with the quality too much. If quality of the topics are of priority then it is better to stick with the base model in order to maximize the coherence and diversity. For archival purposes, it might be preferable to ensure full coverage, but to allow for efficient retrieval for the end user, too much coverage may become counterproductive and decrease the usefulness of topics.

== Heavy preprocessing vs Semantic Structure <disc:pproc>

This study also aimed to investigate the difference between different levels of preprocessing when generating the embeddings of BERTopic and CTM. The models where constructed with the aim of being able to utilize the semantic context of datasets instead of spending effort to clean up and remove stopwords in datasets. This comparison was inspired by the study by @murugaraj-etal-2025-mining where they notices that BERTopic appeared to perform better when the data was more preprocessed.

The results of this study strengthen the claims that BERTopic may benefit from a more preprocessed dataset. Its coherence greatly benefited from using more heavily preprocessed data to generate its embeddings. This shows that it is able to extract semantic relationships even without the stopwords that are usually seen as important for binding text together, and does so more effectively.

This improvement might be explained by the fact that the OCR noise is so high that it disrupts the semantic flow to the point where removing the majority of the words actually leads to an improvement. This will let the models focus on the more context heavy words instead of getting distracted by potential noise or spelling variations. 

Interestingly however, when using the BERTopic version with modified @hdbscan parameters, using the more preprocessed data actually decreased the coherence score of the model, while the version using the less preprocessed dataset was less affected. This may indicate that the stopwords play an important role when differentiating more noisy and vague documents.

when using the reassignment strategy for BERTopic, the effects are not so clear, as the coherence score is more varied across number of topics and preprocessing level. No clear pattern emerges as the heavily preprocessed embedding sometimes perform better, sometimes the same, and sometimes worse.

In the case of the CTM, no meaningful difference was able to be observed between using @d1 and @d2 for the embeddings. There was no benefit but no drawback either. This suggests that the CTM is limited by other factors such as the noise in the dataset or the historical context and the change in embeddings make little difference in comparison. 

The limited effect on CTM may also be explained by that not enough heavy preprocessing was not done. The stopwords removed was based on a single stopword list for modern Swedish that was combined with a list generated by a large language model based on the aforementioned list and words found in the corpus. The stopword removal in the study was very limited and arguably flawed, so the fact that such a large difference was able to be observed in BERTopic is very noteworthy.

Overall it is surprising that removing the stopwords that play an important role in keeping the semantic structure of a text actually can benefit a model designed take advantage of them. This shows the contextual power that can be stored in embeddings and how they can extract meaning that does not always make sense by human merits but still produce powerful results.

== Optimal Number of Topics

The optimal number of topics is not a conclusive result as multiple values for $K$ all showed similarly high scores. Considering the variance in the scores of BERTopic across multiple runs, more runs would be required to confidently proclaim the best suited $K$.

BERTopic and CTM appeared to score slightly lower for $K = 10$ suggesting that 10 topics might be to small to properly explain the dataset, but there is no notable drop off in coherence as topics increased. This may suggest that the dataset is very rich in topics and may hint at value in evaluating the dataset at higher number of topics than just 60. 

In terms of diversity, as described in section @disc:diversity  most models did show a drop off in diversity as the number of topics increased, which would suggest that a lower amounts of topic would be preferable to use. With the sole exception of the ETM, the drop off is not catastrophic and still without acceptable bounds. 

LDA show almost no drop off at all and the CTM is staying rather stable although there is a small decrease. BERTopic does show a non negligeable drop off, it is not problematic in the evaluated range but if the trend continues it would probably not be meaningful to use past a certain number of topics.




== Practical Implications for Riksarkivet <disc:practical>

For Riksarkivet, the results suggest that topic modeling is a promising approach for organizing OCR-transcribed historical Swedish documents. using the standard BERTopic appears to be the most suitable model among those tested, because it combines the highest coherence with reasonable runtime and diversity.

Based on the metrics the results are competitive with studies performed on other datasets. This includes both other historical datasets as well as contemporary and well behaved datasets without OCR noise.

However, the results also suggest that topic modeling should be used carefully. The quality of the topics depends on preprocessing, noise handling, and the number of topics. In a practical archive system, the model should probably be combined with manual inspection or domain expert feedback before being used for public-facing retrieval or browsing. It may also be useful to run the model multiple times to maximize the results due to the stochastic nature of BERTopic.

Before applying the models, more advanced stopword removal should be applied. This study only performed stopword removal in a limited form. Putting more effort into it should not only produce higher results on the metrics, but also provide cleaner topics overall, as the current  topics contain a large amount of noise, and unimportant words.

The findings also indicate that models should not only be evaluated by coherence and diversity. Since the final goal is human use, future applications at Riksarkivet would benefit from human evaluation, especially by archivists or researchers familiar with the document collection.



== Summary of the Discussion <disc:summary>

In summary, the discussion show that BERTopic gives the overall best performance when considering coherence and diversity together. This is followed by CTM, LDA and finally ETM. If diversity is of greater importance, then CTM is instead the strongest model.

If full topic coverage of the data is required, BERTopic can be modified to reassign all unassigned documents to the to the most fitting topic. This mainly comes at a cost of a lowered diversity but still keeps a the high coherence of the base model. If the loss in diversity is considered too high, the CTM is instead the better alternative to use.

The use of the more heavily preprocessed @d2 embeddings showed clear improvements in both coherence and diversity for the base BERTopic model. These patterns were however not clearly visible in any of the other models, with the CTM showing virtually no difference between the @d1 and @d2 embeddings.

The discussion shows that the evaluation results depend on the intended use of the topic model. The base BERTopic model produced the strongest intrinsic results, while the reassignment variant provided complete document coverage with only a moderate loss in diversity. CTM remained important because of its high diversity, and LDA showed that classical models can still be competitive on this dataset. These findings indicate that the best model is not determined by a single metric alone, but by the balance between topic quality, diversity, runtime, and document coverage.
