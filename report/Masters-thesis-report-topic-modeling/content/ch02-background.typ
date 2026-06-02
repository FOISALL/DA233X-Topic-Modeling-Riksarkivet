= Background <bg>

This chapter provides insights into the field of topic modeling as well as some auxiliary fields within @nlp that will be relevant for some models or the data.

@bg:tmo gives an overview into the field of topic modeling while @bg:om describes some of the foundational models and introduces the many shortcomings topic modeling has struggled with. @bg:lda through @bg:bertopic goes elaborates on all the models that will be considered for this study by explaining overall how they work and how that effects their strengths and weaknesses. @bg:mtr introduces the main evaluation metrics that will be considered in this study.

@bg:bert and @bg:sbert introduces BERT and @sbert which is the neural foundation of the most modern models used in this thesis.

Section @bg:ocr gives an overview of what OCR-transcription is and what problems this generally causes for topic modeling.

@bg:rw compiles multiple studies on how topic models have been used in the contexts of noisy, multilingual and historical data. @bg:mbert describes how BERT, which is the neural foundation of more modern models fairs in multilingual contexts. @bg:cstuds describes overall how some models have compared to eacother on mostly unproblematic datasets. @bg:noisydata reviews how previous studies handles noisy datasets for use in topic modeling. @bg:swedata summaries a few studies that has applied topic modeling specifically on Swedish datasets and the problems specific to Swedish. @bg:histdata finally summarizes studies that has applied topic models to datasets that spans larger timeframes and are of historical nature.

== Topic Modeling <bg:TM>

=== Topic Modeling Overview <bg:tmo>

In this modern age, there is a massive amount of data stored digitally. This has created a great interest in the research field of information retrieval. Within this field there is a great need for summarizing large amounts of texts in some kind of lower dimensional space. One popular method of summarizing text is topic modeling. Topic models map the textual data to lower dimensional subspaces that are represented as topics. Essentially, documents are assigned "topics" that should represent their most prominent themes based on their content. Though there are many different models for this, they all try to uncover some latent semantic structures from the text collections @chauhan_shah_2022 .

Most models use the collection of words in each document to decide which topics they should be classified as. Usually a probabilistic distribution is used for this. Each topic is defined as a mixture of words, each word in the vocabulary has a certain probability for appearing in each topic. For example, the word "Bank" may have a high probability to appear in a topic called "River" and "Finance" but lower probability to appear in a topic called "Furniture". In the same way that topics are defined as a distribution over words, the documents themselves are represented by a distribution over the topics. Every document has a certain probability to belong to every topic based on the words they contain. @chauhan_shah_2022 @vayansky_kumar_2020.

Most models are also generative. This means that they don't just assign a probability to the words and documents, but actually attempt to model the entire distribution so that they can potentially generate documents with similar word distributions@vayansky_kumar_2020.

The topics are generated in an unsupervised manner. This means that the content of the topics is learnt without manual labeling. The final topics are not known beforehand, but the number of topics is in many models treated as a hyperparameter @sharma_kumar_chand_2017.

Most Traditional models only consider word frequencies, this is known as the @bow model. The BoW-model means that the order and overall context of the words are ignored. Observing what words, and how many of them are present is often enough to achieve sufficient results @abdelrazek_eid_gawish_medhat_hassan_2023.

Recently there has been a shift from the classic probabilistic models to the more novel @ntm:pl. A compelling advantage with @ntm:pl is that they don't require the same complex mathematical derivations as the probabilistic models, instead replacing it by a neural architecture. This comes with the disadvantage of less interpretability as the latent embeddings of a neural network is not easily understood. Neural models are much more flexible when it comes to the kinds of data they can handle and some are much more scalable compared to traditional models. Additionally, many @ntm:pl can take context into account instead of just focusing on frequencies of words@abdelrazek_eid_gawish_medhat_hassan_2023.


=== Older Models <bg:om>


==== Tf-idf scheme <bg:tf-idf>

Before the large surge of the probabilistic models, research focused more on the algebraic approach. They used term frequencies and classical dimensionality reduction@abdelrazek_eid_gawish_medhat_hassan_2023.

The first implementation that that paved the way for topic modelling was the tf-idf scheme from 1983 @salton_1983 . Here the term frequency (tf) and inverse document frequency (idf) are used to get an estimate of what a document is about by highlighting the most important and unique terms. the tf-value for a term "t" is calculated by counting the number of times t is used in a document "d" and dividing by the total number of different terms in the document. If $f_(x,y)$ represents the frequency of the term x in document y it is given by the formula
$
"tf" = f_(t,d)/(sum_(t' in d)(f_(t',d))).
$

the idf value for a term is then calculated by first counting how many documents in the corpus contains the term at least once, to yield its document frequency (df). The df value is then used to calculate the idf using the formula 

$
"idf" = log(N/"df")
$ @dillon_1983.

Once the final tf-idf score is obtained by multiplying the tf and idf values, we can see what terms rank the highest for every document. The overall rare terms that are more common in the specified document rank the highest. This also allows us to store the documents more efficiently by storing the scores in a matrix where the document has a score for every term. Instead of storing all the documents with all their terms we just store a T-by-D matrix, where T contains the tf-idf value for each term and D represents all the documents. The reduction is however rather insignificant as the matrix is very sparse as most scores are zero, since most words of the vocabulary are not in every document. Additionally the results didn't yield much information about the relationships between documents.@vayansky_kumar_2020.

==== LSI <bg:lsi>

A major improvement from the tf-idf scheme was the @lsi developed in 1990. The major contribution of the LSI was to use the matrix generated by the previous method, and perform @svd on it @vayansky_kumar_2020. This allows the model to not just count words, but capture broader concepts. @svd splits a matrix A into 3 matrices @sauer_2014

$
A = U Sigma V^T.
$

U contains the "left singular vectors" which are the eigen vectors of $A A^T$ which is a T-by-T matrix, relating terms to eachother. Thus U relates the terms to the concepts. Similarly, V contains the "right singular vectors" which instead are the eigen vectors of the D-by-D matrix $A^T A$, making V instead relate the documents to the concepts. Finally $Sigma$ contains square roots of the eigenvalues of $A^T A$ and $A A^T$. Both matrices share the same eigenvalues since they are symmetric. These eigen values represent the concepts or "topics" in our case and is what allows us to make the connection from the terms and documents to the matrix. This is the first version of something that can be described as "topics". This also lets us disregard the eigenvalues which have a very small value, thus letting us compress the data by moving away from focusing on individual words to focusing on a smaller number of broader topics  @deerwester_dumais_furnas_landauer_harshman_1990.

@lsi solves the issue of synonymity in the simple tf-idf scheme. Now 2 terms having the same meaning will point to the same concept. It was also able to handle a degree of noise. Since only the larger eigenvalues are kept, weaker relationships generated by a few misspelled terms were able to be disregarded @abdelrazek_eid_gawish_medhat_hassan_2023.

@lsi however has some major drawbacks, most importantly, it was not able to handle words with multiple meanings since every term only has a single point in the concept space. Its algebraic nature also makes the topics less interpretable to humans. This is a common problem of @svd.

The fact that it requires the handling of very large matrices also makes it impractical for larger datasets.

==== pLSI <bg:plsi>

Some of these problems were attempted to be solved by its successor, the @plsi from 1999 @hofmann_2017. the @plsi uses a statistical foundation to create a generative model, allowing it to model a more generic semantic structure@vayansky_kumar_2020.

Like the modern models, @plsi is unsupervised but still requires knowledge of how many topics should exist. Much like the @lsi model, we still need the term-document matrix. the @plsi handles 3 main variables: z being the topics, w for a term, and d for a document. It focuses on the 2 probabilities $P(z|d)$, the probability of each topic for every document, and $P(w|z)$, the probability for each word appearing in every topic. Based on these probabilities we can use Bayes' theorem to calculate the probability that topic z explain term w in document d

$
P(z|w,d) = (P(z|d)P(w|z))/(sum_(z'in Z) P(z'|d)P(w|z')).
$

This probability is then iteratively maximized using the Expectation-Maximization (EM) algorithm@hofmann_2017.

What we are left with in the end is the probabilities of each word appearing in each topic, and the probabilities of every topic being connected to each document. The latter being the problematic case. Since the probabilities must be stored once for every document it scales linearly with the number of documents, making it ill suited for a large corpora. Additionally, there is no way of easily integrating a new document to the distributions without redoing the whole algorithm@vayansky_kumar_2020 @blei_edu_ng_jordan_edu_2003. Despite its shortcomings, it introduced many of the core concepts still used in modern topic modeling today and solved some of the problems with previous models. The model is able to handle Polysemy, meaning that words can now have different meanings. This is made possible by two different topics now beign able to have a high probability for the same term. Probabilistic models have been the main focus of topic modeling up until recently when neural models have started entering field as well @abdelrazek_eid_gawish_medhat_hassan_2023.

=== LDA <bg:lda>

Latent Dirichlet Allocation (LDA), introduced in 2003 has been and still is one of the most widely used topic models @blei_edu_ng_jordan_edu_2003. LDA solidified @bptm:pl as the standard modeling approach by addressing many of the shortcomings present in previous models such as @plsi. @plsi's approach to tracking topic mixture per document "$p(z|d)$", was flawed as it scaled poorly and didn't allow for new documents to be integrated after training. LDA went around this issue by defining a universal distribution from which all documents are assumed to have drawn their topic mixture, namely a Dirichlet distribution @chauhan_shah_2022 @vayansky_kumar_2020.

LDA uses a 3-level hierarchical Bayesian structure with parameters on corpus level, individual document level and individual word level@blei_edu_ng_jordan_edu_2003. 

*Corpus level: * The Outer layer defines 3 Hyper parameters, $alpha, beta "and" k$. $alpha$ is the Dirichlet prior for the document-topic distributions. It controls the sparseness of topics in the documents, a low value leads to documents with fewer topic associations. $beta$ is the Dirichlet prior of the topic-word distributions. Similarly to $alpha,$ a lower value for $beta$ corresponds to a lower amount of words associated with each topic. Lastly the hyper parameter $k$ is the number of total topics, also present in most other topic models.

*Document level: * The document level parameter is $theta_d$. This is the parameter which gives the whole topic distribution for a specific document $d$. In a corpus of 3 topics this vector might look like $theta_d = [0,3, 0,6, 0,1]$ meaning it's represented 30% by topic 1, 60% by topic 2 and 10% by topic 3. This is drawn from a Dirichlet distribution parameterized by $alpha$ 
$
p(theta|alpha) = (Gamma (Sigma^k_(i=1)alpha_i))/(Pi^k_(i=1)Gamma(alpha_i))theta^(alpha_1 -1)_1 dot ... dot theta^(alpha_k -1)_k.
$

*Word level: * The word level parameters are $z_(d n)$ and $w_(d n)$. The parameter $z_(d n)$ is the specific topic assigned to word number $n$ in document $d$, sampled from $theta_d$. The actual word in the text is $w_(d n)$, which is assumed to have been sampled from $beta$, indexed by $z_(d n)$.

The only parameter we are actually observing is $w_(d n)$, from which we will then infer the latent variables $z "and" theta$. This is given by the equation 
$
p(theta, z |w,alpha,beta) = (p(theta, z, w| alpha, beta))/(p(w|alpha,beta))
$
where the denominator $(p(w|alpha,beta))$ is intractable. Though it cannot be calculated exactly it can be approximated using techniques such as variational inference or Gibbs sampling@blei_edu_ng_jordan_edu_2003.

#figure(
  image("LDAstruct.png"), caption: [Graphical model of the LDA (from Figure 1 on page 5 of @blei_edu_ng_jordan_edu_2003)]
)


This makes LDA into a fully generative model, allowing it to apply the universal Dirichlet distribution to new, previously unseen documents. Despite this, LDA still has several drawbacks. It still heavily relies on the @bow model, preventing it from doing more than just counting the words, completely ignoring their semantic context. LDA also struggles with vocabularies containing many rare words or overrepresented words such as stop words. Additionally, even if it has the ability to handle new documents, they have to be similar to the existing documents and cannot handle a document in a new language @dieng_ruiz_blei_2020 @abdelrazek_eid_gawish_medhat_hassan_2023 @bianchi_terragni_hovy_nozza_fersini_2021.



=== ETM <bg:etm>

The ETM was introduced in 2019. It was designed to be able to handle much larger datasets than the classic LDA. It is also able to handle a corpus containing many rare words as well as stop-words; some of the biggest drawbacks of the LDA. It achieves these benefits by instead of representing each word as discrete symbols, it uses embeddings to represent both words and topics as coordinates in the same embedding space @dieng_ruiz_blei_2020. Though it was initially described as a probabilistic model with extensions, in context of the more modern models, it is now considered to be a @ntm by todays standards @abdelrazek_eid_gawish_medhat_hassan_2023 @wu_nguyen_anh_tuan_luu_2024. 

#figure(
  image("ETM_embedding_space.png"), caption: [embedding space showing topics about sports (From Figure 3 on page 3 of @dieng_ruiz_blei_2020)]
  
)



It uses a similar generative process as the LDA in @bg:lda but instead of using a Dirichlet distribution, a Logistic-Normal distribution is used. This distribution will be easier to optimize later on using the reparameterization trick @dieng_ruiz_blei_2020.

The ETM attempts to optimize both the word embeddings $rho$ and topic embeddings $alpha$ to best explain the observed words in the document. This is done by maximizing the marginalized log-likelihood for all documents. The equation 

$
cal(L)(alpha, rho) = sum^(D)_(d=1) log(p(w_d|alpha,rho))
$
can be interpreted as "what embeddings for words and topics ($alpha$ and $rho$) gives the highest probability of generating the words "$w$" in all the documents "$D$". Similarly to the LDA, this probability is intractable. The reason for this is that calculating 

$
log(p(w_d|alpha,rho)) = integral(p(delta_d) product^(N_d)_(n=1)p(w_(d n)|delta_d,alpha,rho))d delta_d
$
requires us to calculate every possible topic mixture for every document $delta_d$ which is not analytically possible.

To get around this, amortized variational inference is used @N_Goodman_S_Gershman_2014. The inference network is an encoder network that is optimized using the Evidence Lower Bound (ELBO) and reparameterization trick, the details of which are omitted as it is outside the scope of this paper@dieng_ruiz_blei_2020.

The result of this is a coordinate vector $alpha$ for every topic as well as a coordinate vector $rho$ for every word. Additionally, we also get the trained encoder. This encoder can now be used to classify the most prominent topics in any given document. 

This solves the previous problem of not being able to account for new unseen documents as the trained encoder can receive any document and apply its already learned embeddings @abdelrazek_eid_gawish_medhat_hassan_2023. It also handles stop words well as it can consider them to make up their own "junk" topic instead of confusing them with other topics @dieng_ruiz_blei_2020.

This make the ETM a strong and scalable model, it can handle vocabularies that are both large and sparse. Despite its advantages, it still suffers from a drawback present in most contemporary models as well, the @bow:pl model @abdelrazek_eid_gawish_medhat_hassan_2023, @bianchi_terragni_hovy_nozza_fersini_2021. This means that it only takes into account frequency of words in the document, not their order or their context @dieng_ruiz_blei_2020.

It also struggles with polysemy, when one word have multiple meanings, this is a symptom of it only having one vector embedding for every word as it will try to push the embeddings vector in two directions at the same time. This is a known problem with word2vec which ETM uses to generate its embeddings @mikolov2013efficientestimationwordrepresentations.



=== CTM <bg:ctm>

The Contextualized Topic Model (CTM), introduced in 2021 is an @ntm that attempts to move away from the classic @bow model @bianchi_terragni_hovy_nozza_fersini_2021. Its purpose is to improve previous models by allowing them to a greater extent handle semantic meaning in text. Specifically the semantic meaning from a text should be able to be considered regardless of what language it is written in. Most previous models were monolingual @abdelrazek_eid_gawish_medhat_hassan_2023. The CTM showed that by partly replacing the @bow model with a neural network, a multilingual model could be achieved @bianchi_terragni_hovy_nozza_fersini_2021 @wu_nguyen_anh_tuan_luu_2024. 

The CTM is built with the framework of a @vae. An encoder guesses the topic proportions, and a decoder reconstructs the documents based on the encoder. The framework is somewhat similar to the ETM described in section @bg:etm, but the input space is drastically different @dieng_ruiz_blei_2020. Changing the input space is the main contribution of the CTM as it the major difference from the neural-ProdLDA, which it can be considered to be an extension of @srivastava2017autoencodingvariationalinferencetopic  @bianchi_terragni_hovy_nozza_fersini_2021.

Instead of calculating the word frequencies like in the @bow model, the document is passed through a pre-trained SBERT model to generate dense, contextualized embeddings @reimers-gurevych-2019-sentence. The encoder inference network use these embeddings as input and return the mean and variance of a gaussian distribution. The final topic proportions are acquired by using to reparameterization trick to add noise to the parameters and applying the softmax function which gives us our 

$
theta_d = mu_d + sqrt(Sigma_d) dot epsilon
$
containing the topic mixture @wu_nguyen_anh_tuan_luu_2024 ,@srivastava2017autoencodingvariationalinferencetopic.

Next the topic mixture is used as input to the decoder. In contrast to the encoder, the decoder actually uses the classic @bow model instead of SBERT. This lets the CTM learn the words in an interpretable way as opposed to the black box that is the SBERT embeddings. First a matrix $beta$ containing the probability of every word existing within every topic. This is used to generate a guess of the real probabilities $accent(x,"^")$ using 

$
accent(x_d,"^") = "softmax"(beta theta_d).
$

This guess is then compared against the real distribution $x$ giving the reconstruction loss
$
cal(L)_"reconstruct" = -sum^V_(v=1)(x_(d,v)log(accent(x,"^")_(d v)))
$
which is then used together with the KL-divergence to form the ELBO that is used in back propagation @wu_nguyen_anh_tuan_luu_2024.

#figure(
  image("CTM-structure.png"), caption: [Architecture of the CTM (From figure 1 on page 2 from @bianchi_terragni_hovy_nozza_fersini_2021)]
)



The training step uses both the encoder and decoder to learn the topic mixtures. After the training is done however, only the inference using the encoder is used and will output $theta_d$ containing the topic mixture for the specific document. Thus, in the inference part the @bow model is not used at all @bianchi_terragni_hovy_nozza_fersini_2021.

This architecture is what allows the multilingual properties of the CTM. By detaching the reading process from the @bow model, the embeddings will be based on the semantic meaning of the documents and not their words. CTM will simply inherit the multilingual properties of SBERT @bianchi_terragni_hovy_nozza_fersini_2021.


The CTM tries to abandon the "Bag-of-Words" model used in the ETM and instead uses BERT to generate the embeddings. This will allow the model to take entire contexts of documents into account when sorting them into topics. Using BERT for embeddings should also allow the model to work better on multilingual data, allowing it to be applied to a wider range of datasets@bianchi_terragni_hovy_nozza_fersini_2021. 

=== BERTopic <bg:bertopic>

BERTopic, introduced in 2022 attempts to tackle the problem of topic modeling from a different angle by using @sbert to generate embeddings. Even though the standard approach has shifted from probabilistic models to @ntm:pl they all still share the same method of treating the topics as a probabilistic distribution which they try to reverse engineer using a generative approach. BERTopic replaces the generative approach by instead treating it as a clustering problem @Maarten_G_2022.

Similarly to the CTM, BERTopic replaces @bow with a pre-trained @sbert model in order to grasp semantic meaning and store entire documents as an embedding vector. Unlike CTM, BERTopic doesn't just partially replace the @bow model but omits it entirely for learning the topics. It only retains it for later on naming the topics.

There are however problems when attempting clustering in high dimensional spaces. Differences in distance between close and distant pairs has been shown to shrink as dimensionality increases, making the concept of locality less defined @lowdimdist. Though there are still methods to perform clustering in these spaces, the authors of BERTopic instead opted to dimensionality reduction in order to circumvent the problem.

The method chosen for dimensionality reduction is @umap. It was chosen over well known methods such as PCA for its abilities to better preserve global and local features @mcinnes_healy_saul_großberger_2018.

Once a low dimensional embedding for the documents is obtained, they are clustered using @hdbscan @mcinnes_healy_astels_2017. This clustering method improves topic representations by disregarding outlier documents as noise, keeping the topic clusters coherent as they are not cluttered with loosely related documents.

Once the clusters are formed, they will each be assigned a topic. instead of just naming the topics using the centroid of the clusters, BERTopic aims for more representative topic naming. BERTopic introduces a modification to the classic TF-IDF procedure introduced in @bg:tf-idf. Instead of calculating the score for a word based on a frequency across all documents, it is calculated across the topics instead. First, all documents in a cluster are concatenated into one large document. Then we find the term frequency $"tf"_(t,c)$for every term "t" in the concatenated document "c". Then the term frequency $"tf"_t$ for the same term is calculated across the whole corpus. The final score of a term is then calculated as 

$
W_(t,c) = "tf"_(t,c) dot log(1 + A/("tf"_t))
$
where A is the average number of words per cluster. The word with the highest score will give name to the topic. In case fewer topics are desired than what was generated by @umap, the least common topics can be merged with the most similar other topic @Maarten_G_2022.

The fact that a clustering algorithm is used means that we can skip the computational overhead that comes with the backpropagation of other models such as CTM and ETM. The topics generated are based on semantic meaning and should be language independent granted that the SBERT model used is multilingual @Maarten_G_2022, @abdelrazek_eid_gawish_medhat_hassan_2023.

A drawback that sets it apart from other models however is that it assumes a single topic for every document, instead of handling it as a mixture of topics. Another limitation is also that even though the @bow model is unused for the actual cluster of the topics, the naming scheme for the clusters is still based on the word frequencies of the documents. This often leads to many of the top rated words in a topic being very similar to the point of being redundant variations of the same words@Maarten_G_2022. 

== Evaluation Metrics <bg:mtr>

=== Topic Coherency <bg:chr>

Topic coherency is an evaluation metric used in topic modeling to get a grasp on how well the documents within the topics actually fit together. Previously, a measure known as perplexity was used. Perplexity measured the how well the model generalizes to unseen data by testing how well the model predicts a separate set of test documents. The perplexity measure was however shown to not match with human evaluations, and topic coherence was created with the aim of bridging this gap. Coherence was meant as a measure to focus more on if the topic semantically make sense. @newman_jey_lau_grieser_baldwin_2010 @lau2014machine. 

Topic coherence uses @pmi to get a statistical value of how often the $N$ most common words in a topic actually appear together frequently in documents, compared to just independently. The @pmi is calculated pairwise for two words "$w_i$" and "$w_j$"as 
$
"PMI"(w_i,w_j) = log(p(w_i,w_j)/(p(w_i) dot p(w_j))).
$
Co-occurrence is only recognized if the words appear within a 10-word sliding window of each other.

@pmi is however not sufficient as it is. The score is unbounded and shows bias towards rare words. Because of these drawbacks, modern implementation instead use @npmi @bouma2009normalized @lau2014machine

$
"NPMI"(w_i,w_j) = "PMI"(w_i,w_j)/(-log(P(w_i,w_j))).
$ 

This measure normalizes the score by dividing the original @pmi with the negative log of the joint probability of the same word pair. This ensures that the score is bounded between -1 and 1 where 1 indicates a strong coherence, 0 means they are independent and -1 means that they never co-occur.

This value is calculated for all $binom(N,2)$ pairs and then averaged @newman_jey_lau_grieser_baldwin_2010 @dieng_ruiz_blei_2020.

Topic coherence is still one of the most used metrics for topic modeling and assigns high scores to coherent and interpretable topics. Coherence has however shown some flaws when it comes to evaluating topics generated by @ntm:pl. The @npmi assumes that the coherence of the topics must be indicated by the co-occurrence of words, which is how probabilistic topic models derive their topics, but @ntm:pl derives them from semantic meaning. This leads to @ntm sometimes scoring lower in coherence even though a human evaluator might have scored them comparatively higher @abdelrazek_eid_gawish_medhat_hassan_2023 @Hoyle. Similarly @ntm:pl has been shown to sometimes score surprisingly high in coherence even though the topics appear incoherent to human evaluators. This is most likely caused by the richness of the embeddings finding connections that are not explainable to humans @Hoyle.


=== Topic Diversity <bg:div>

Topic diversity is a metric that aims to promote topic categorization with very distinct topics. While topic coherency described in @bg:chr ensures the topics are interpretable and actually share a common theme, topic diversity ensures that the different topics aren't all the same @abdelrazek_eid_gawish_medhat_hassan_2023.

The diversity score is trivially calculated by counting the amount of unique words in the first $N$ words of every topic $k$.
This value is then divided by the maximum number of unique words, meaning $N dot K$ @dieng_ruiz_blei_2020. A low score means that there are many redundant topics and a high score indicates that the model has successfully generated a set of distinct topics unique from each other @dieng_ruiz_blei_2020. lowest score of 0 means that all topics share all their top words, and the maximum score of 1 means that none of the topics share any words at all. Increasing the number of topics too far usually drops the score as the model forcibly divides the corpus into more topics than what is appropriate @abdelrazek_eid_gawish_medhat_hassan_2023.

== BERT <bg:bert>

Bidirectional Encoder Representations from Transformers (BERT) is a language representation model introduced in 2018. It's meant to be used as a pretrained language model that can be used by another @nlp model. Much of its claim to fame comes from its ease of use when used for fine tuning and being adapted to different domains. Its main contribution however is the fact that it is bidirectional as opposed to unidirectional. Many models before BERT were unidirectional, such as the well known Generative Pre-trained Transformer (GPT) @devlin_chang_lee_toutanova_2019. 

A unidirectional model only reads the text from one direction, this means that it only learns to predict a word based on its predecessors, not what comes after. BERT aimed to remedy this shortcoming by taking both preceding and succeeding context into account. This is done by first using an encoder transformer that can perceive an entire document at once. Then it is trained by using a @mlm. The @mlm randomly hides (masks) tokens in the input and sets the objective to correctly predict the missing tokens based on its context@devlin_chang_lee_toutanova_2019. 

BERT also uses "next sentence prediction" where the model is trained on pairs of sentences from the corpus and is tasked with deciding which pairs are true pairs, present in the corpus and which pairs are randomly assembled@devlin_chang_lee_toutanova_2019.

This allows BERT to perceive context to a much greater extent compared to many of its contemporaries such as word2vec that struggles with polysemy @gardazi_daud_malik_bukhari_tariq-alsahfi_bader_alshemaimri_2025 @mikolov2013efficientestimationwordrepresentations  or @bow that disregards semantics entirely@Mikhail_Koroteev_2021 @dieng_ruiz_blei_2020.

The pretrained model can be fine tuned to create highly accurate and contextualized embeddings. The embeddings can be used in a plethora of @nlp applications such as Grammar error detection, cross-lingual transfer learning, tokenization among many other tasks. @gardazi_daud_malik_bukhari_tariq-alsahfi_bader_alshemaimri_2025, @Mikhail_Koroteev_2021. This fine tune is done by either @mlm or the more modern approach approach, @tsdae @wang-etal-2021-tsdae-using.

The @tsdae methods produces better embeddings for the document as a whole compared to @mlm. It works by encoding a document with a large portion of masked out words into an document embedding and is then trained by having a decoder attempt to reconstruct the unmasked document. By encoding the whole document it forces a much wider context into the embeddings compared the @mlm approach @wang-etal-2021-tsdae-using. 

Despite these prominent advantages, BERT still come with some weaknesses that somewhat limits it. BERT uses tokenization to perceive texts but it has a hard limit of 512 tokens @gardazi_daud_malik_bukhari_tariq-alsahfi_bader_alshemaimri_2025. This limits the scope of its context perception which will affect its performance on larger documents. It requires extensive GPU resources to train the model, making it infeasible for most actors. However since most use cases only focus on fine tuning the model, which still may require GPU performance but significantly less, this is often not a problem @Mikhail_Koroteev_2021. Another one of its weaknesses is in the domain of Semantic Text Similarity (STS) where it is very inefficient for finding the 2 most similar sentences in a corpus, leading to massive computational overhead @reimers-gurevych-2019-sentence.

== Sentence-BERT <bg:sbert>

@sbert was designed in 2020 with the intent of extending the original BERT model (described in @bg:bert) to domains it was initially unsuited for. Specifically, BERT struggles with comparing text. If BERT was tasked with finding the most similar pair of sentences in a corpus it would have to run the inference once for every possible pair. This is great for getting a deep and accurate comparison but causes a time complexity of $O(n^2)$ making the task impractical for larger scale tasks given the architecture. This extends to similar unsupervised tasks such as clustering @reimers-gurevych-2019-sentence.

@sbert remedies this significant limitation by ensuring that the sentences can be inputted individually to generate an embedding. This embedding will preserve semantic meaning in a way that allows it to be compared to the other embeddings using cosine similarity @Maarten_G_2022. This way the comparison step has been moved outside of the network. This has been shown to lower the computation time of comparing 10. 000 sentences from 65 hours with BERT, to a mere 5 seconds using @sbert. This comes at a cost of slightly decreasing its capabilities of performing deep comparisons and might sometimes miss less obvious similarities. This is regarded as a well worth trade-off to be able to leverage the capabilities of BERT in previously unavailable domains @bianchi_terragni_hovy_nozza_fersini_2021 @reimers-gurevych-2019-sentence.

== OCR <bg:ocr>

@ocr is the process of scanning textual images into actual text data. This is often used so that the textual data can be further processed digitally @mori. @ocr has a wide range of use cases such as transcribing text from phone cameras for instant translation, reading traffic signs and road names in autonomous vehicles and transcribing handwritten text to preserve legal or historical documents.

Traditional methods use various multi-step processes that involve separating the text from the background, isolating text from noise and formatting such as de-skewing and other morphological operations @hamad_kaya_2016. Newer methods apply deep learning approaches to simplify the process. Examples of improvements include using convolutional networks for isolating the text and auto-encoders for interpreting it @long_he_yao_2020.

Despite improvements in @ocr technologies they are still prone to including errors in their output. The quality depends greatly on the input text. factors such as lighting, text orientation and fonts all affect the end result @hamad_kaya_2016. The most common errors are character substitution and spacing errors. An "o" can easily be mistaken for a "0" considering variations in handwriting. Spacing between characters can also be mistakenly created or ignored, interpreting "of the" as "ofthe" or "project" as "pro ject" @hamad_kaya_2016, @Lopresti_Daniel. These errors can cause significant problems in downstream tasks depending on their scale and the context.

The quality of @ocr transcribed text can be evaluated using @cer. This metric measures what percentage of characters were transcribed incorrectly. @cer is calculated based on the minimum required insertions "$i$", substitutions "$s$" and deletions "$d$" to transform the original document into the @ocr transcription. The formula is given by 
$
C E R = (i + s + d)/n
$
where $n$ is the total number of characters @neudecker.

Small error rates are usually manageable in the context if classical information retrieval such as keyword searches. In the context of @nlp even small errors on the scale of around 5% has shown to lead to detrimental results in downstream tasks @Lopresti_Daniel.






== Related Works <bg:rw>


=== Multilingual BERT  <bg:mbert>

One of the main contributions BERT claims over other models for @nlp is its simplistic fine-tuning. Instead of fine-tuning all of its parameters, only the last output layer is used for fine-tuning@devlin_chang_lee_toutanova_2019. This has been utilized by many to adapt the BERT model to different domains such as biomedicine, finance and clinical data @gardazi_daud_malik_bukhari_tariq-alsahfi_bader_alshemaimri_2025. Most notably BERT has been fine-tuned to work in many different languages. This raises the question on how comparable the results of the fine-tuned model are to a model working on the domain it was pretrained on. This was investigated in a study from 2020 @nozza2020mask. The results of the study showed that if properly pre-trained, the model consistently showed better results than if just fine tuned. The sole exception to this is when training on low resource languages. In these instances, the data is insufficient to properly pre-train the BERT model and a fine-tune of a more general model thus provide better results @nozza2020mask. 

This problem with general models was also noted by a Swedish study from 2023 @haffenden2023making where they argued that pre-training a model on a large set of languages causes smaller languages like Swedish to be underrepresented. They proved this by training a monolingual model on only Swedish text which consistently outperformed more general BERT models on downstream tasks@malmsten2020playingwordsnationallibrary. They also made a point to mention that they did not only train the model on clean data. They made sure to include some messy data from OCR-transcribed documents to ensure the model got a complete view of the Swedish language @haffenden2023making.



=== Comparative studies <bg:cstuds>

In the last decade there has been much research into the domain of topic modeling and many different models have been developed with many different strengths and weaknesses. Thus, much effort has also been put into evaluating the models in comparison to each other. One of these studies done by Abdelrazek et al. @abdelrazek_eid_gawish_medhat_hassan_2023 divides the topic models into the domains algebraic, probabilistic and neural. They then picked out a few models in each category and compared them to each other. in the algebraic category they used NMF @lee_seung_1999 and LSI, for probabilistic they used LDA and HDP, and for neural models they chose ETM and CTM. This study in particular used two mostly well behaved, english datasets to compare the models. they used _20newsgroups_ and _M10_, which are commonly used in the domain as benchmark datasets. _M10_ is of similar size to _20newsgroup_ but contains significantly shorter documents with 5.91 average words per document and more rare words, instead of the 48.02 in the _20newsgroup_ dataset.

They used the @octis framework to evaluate the models @terragni-etal-2021-octis. The framework provides access to many  common topic models as well as automatic evaluation using many of the common evaluation metrics such as diversity and coherence.

The results of the study showed that in the _M10_ dataset, though @ntm:pl performed slightly better in topic coherence, algebraic and probabilistic models still showed similar results. In topic diversity the prominence of neural models was more noticeable, except for the ETM which performed very poorly. On the _20newsgroup_ dataset the algebraic and probabilistic models performed better and were sometimes even outperforming the neural models. 

The conclusion drawn from the study was that even though algebraic and probabilistic models sometimes perform better than neural models, neural models often outperform the other types as soon as the dataset becomes less well behaved, more sparse or just very large. This has been reflected in the current research trends shifting to focus on neural models instead of probabilistic which previously were the most prominent in the field.


=== Studies on Noisy Datasets <bg:noisydata>

Topic models are often evaluated under ideal conditions with common benchmark datasets. This does however seldom represent the real conditions under which the models are to be used. Often the data sets may be be cluttered with grammatical errors or misspelling. Many studies have thus made an effort to evaluate topic models on specifically this type of data.

LDA was explicitly investigated in a study by Geeganage et al. @geeganage on noisy datasets and evaluated using both coherence and perplexity. The conclusion was a strong relationship between clearly defined topics and the quality of the input of the generated topics. When the LDA was used on noisy data it often generated meaningless topics. 

These problems are echoed by Denecke et al. @denecke2010topic in a study that attempts to remedy these problems on a weblog dataset. They  investigated the LDA but attempted to normalize the dataset by filtering out everything but the proper nouns. They argued that verbs, adjectives and adverbs often are unsuited to describe the topic of a sentence. The study proved that the LDA indeed was capable of correctly assigning topics to noisy dataset, granted that the dataset had been heavily preprocessed.

Often text are extracted from historical or otherwise handwritten text, which is often done using OCR described in @bg:ocr. A study by Snickars @Snickars03072022 used LDA on an OCR-transcribed dataset of Swedish government data containing around 3000 documents. By utilizing stop word removal and stemming to concentrate the vocabulary into only the most essential words he was able to extract meaningful topics.

More recently the problem of noise in the dataset has been tackled by embedding based models as a more robust solution, replacing the probabilistic models with @ntm:pl. This topic was investigated by Egger et al. @egger in a study where they compared the results of LDA and BERTopic among other models on a dataset of Twitter posts. The results show that though all models were able to generate somewhat relevant and meaningful topics, LDA still struggled compared to BERTopic and other embedding based models. LDA generated many broad and less meaningful topics. BERTopic on the other hand was able to generate more specific topics which provided useful insights into the data. Even compared to other embedding models BERTopic generated more diverse topics with little overlap. BERTopic also achieved this without requiring the heavy preprocessing in the form of stopword removal and lemmatization that LDA demands. The authors do however admit that this leads to some redundancy in the top key words in a topics, such as “travel bubble” and “travelbubble," and conjunction words (e.g., after, before, to, from, at) but argues that the content mostly of a topic is still enough to determine its meaning in spite of this.

=== Topic Modeling in Swedish Data <bg:swedata>

Most research in the field of topic modeling focuses on contemporary English corpora. This leads to their performance being under evaluated on texts of different languages. A study that attempts to investigate the performance of topic modeling on specifically swedish data was conducted by Blad and Svensson @svensson2020exploring. They applied LDA and NMF on a dataset of Swedish news articles. The study similarly to previous studies concluded that preprocessing the data was necessary to obtain meaningful results. Despite these efforts however, it still described the results as unsatisfactory and that it could only work as an inferior addition to manual categorization. The dataset being in Swedish instead of English was attributed some of the blame for the poor results. The study refers to Hedlund et al. @HEDLUND2001147 describing Swedish as different from English in the following five areas:

#enum(indent: 2em,spacing: 1em,[
  *A more complex morphology compared to English:* Some examples of this is that Swedish has multiple different plural suffixes, definite forms and inflections forms that often require morphological anlysis programs in order to normalize: If not normalized properly by lemmatization this can cause a noun to be underrepresented as its presence is spread out through its inflections.
],[
  *Gender features:* Swedish nouns belong to one out of two genders, en-words or ett-words. This has a beneficial impact on @nlp as it may guide models as to what a sentence may be referencing back to instead of just referring to everything as "it". However it does nothing to help @bow based models as they cannot consider this context.
],
[
  *Derivations:* Many words in Swedish are often derived from another "base word", for example "lära" can become "lärare". This causes similar problems to the morphological inflections where a single concept become spread out on multiple similar words, causing sparsity in the data
],[
  *Compound words:* Instead of using phrases of multiple nouns, Swedish is characteristic for using compound words. These words usually needs to be decomposed in preprocessing but this is not always straightforward. Some compound words modify their components, complicating the process.
],[
  *Homonymy and polysemy:* Polysemy meaning two words that mean the same thing, and Homonymy meaning one word that has multiple meanings, can together be referred to as Homographs. This ambiguity often leads to confusion is @nlp. This phenomenon is more common in Swedish than in many other languages, with an estimated 65% of words in running texts being homographs, compared to just 50% in English. These circumstances has created reason to believe that Swedish datasets will benefit more from @nlp models that can take context into account than English datasets since this will let them handle the homograph problem better.
])


The study by Blad and Svensson @svensson2020exploring attempted to get around these problems by aggressive lemmatization and also filtered the dataset to only include common nouns. As mentioned before, this worked and did indeed produce sensible results. the quality did however leave much to be desired as it by itself did not provide very deep insights into the dataset that wasn't obtainable through manual means.


Another study by Bernadeta et al. @Bernadeta_2023 used topic modeling to investigate a dataset of SVT published articles about  the Covid pandemic. This study used LDA, Dynamic Topic modeling (DTM) and BERTopic. To fully be able to utilize the @bow representation they attempted stemming to preprocess the data, but similarly to @svensson2020exploring they realized that this was not sufficient, referencing problems with word suffixes. Instead they opted to use the more advanced lemmatization tool stanza. This finally made the LDA viable for the dataset.

In order to find the optimal number of topics to generate they ran the model multiple times, using the number of topics as a hyper parameter. The study chose to use manual evaluation instead of intrinsic metrics such as coherence scores. They don't deny the usefulness of such metric but describe human evaluation as sufficient.

Though the paper mostly focused on analyzing the results of the LDA and DTM, they mention how it is worth exploring neural alternatives to the probabilistic models. They attempted to run a BERTopic model on the dataset which yielded sensible and interpretable clustered topics, showing that BERTopic works on swedish data, unfortunately they did not specify how or if they finetuned the underlying @sbert model. The topics were also able to be plotted over time, showing when each topic had the most articles posted.

=== Topic Modeling in Historical Data <bg:histdata>

Historical data poses multiple different challenges. Since it originates from a different time frame the language used is often different from modern language. Additionally, if the historical data comes from a very wide time frame there may also exist significant language differences even within the dataset. To further complicate matters, since the historical data often originates from a time before digital storage it is not uncommon that large parts of the data have been transcribed using OCR and thus also come with many of the problems mentioned in @bg:ocr.

This problem was investigated by Hall et al. @hall_mernitz_rensch_2026 in a study where they tested topic models on a datasets of 19th century German and one of Ancient Greek corpora. The study used mainly LDA and found that most of the problems stemmed from the ability to properly preprocess the data to fit with the @bow representation. They concluded that even though both corpora were filled with OCR errors, they were not very prominent in the final topics. Instead the lemmatization was the bigger problem since modern software tools are not able to properly handle historical and multilingual variances. They also noted that Greek is a very context dependent language, making it a bad fit for the LDA. The results on the German dataset was however better, generating more interpretable topics according to human assessment. Although the paper points out that the there was a very low degree of inter accessor agreement, questioning the validity of conclusions that can be drawn from such evaluations.

Greek showed the best result when  10-20 topics were generated, whereas German instead performed best on 70 topics. This showcases the need to vary the number of topics as a hyper parameter to find the best results. 

Another study by Bodell et al. @Hurtade_2024 analyzed a large Swedish dataset spanning 1945-2019. They focused on the issue of "language drift" where words change meaning or spelling over time. They describe how this causes a theme to be split over multiple topics as the vocabulary changes over time. Another problem they bring up is how the topics may also change over time and how it is up to the researchers to identify the theme of the topics once they are generated. This leads to uncertainties and may also introduce elements of bias. To get around this problem while still leveraging the effectiveness of the LDA, they modified it into a semi-supervised seeded LDA. By giving the model a set of "seed words" and forcing the model to create topics with words that co-occur with those words. They used both old and new versions of the same words to force words co-occurring with both of them to belong to the same topic, instead of being fragmented across multiple topics. This showed good performance but still struggles with the problem of polysemy. 

In order to achieve more useful results it would make sense to attempt to utilize the more modern @ntm:pl. This was specifically investigated by murugaraj et al. @murugaraj-etal-2025-mining in a study where, similarly to the study by Eggar @egger, they compared the classical topic modeling methods LDA and NMF to the newer embedding based methods Top2Vec and BERTopic. This study however aims to focus on specifically historic data. Their dataset consisted of newspaper articles from 1955 to 2018. They mainly used topic coherence and diversity for evaluation.

Their findings show that NMF creates more coherent topics compared to LDA, while LDA excels in diversity. Top2Vec appeared to struggle with the large quantity of data and performed worse than both NMF and LDA, showing that the flexibility of embedding based models is not always a cheat code to better results. BERTopic however outperformed all three models, scoring higher in both coherence and diversity. Additionally, BERTopic also scaled significantly better with the number of topics generated. While the classical models grew significantly in computational cost with the increase in topics, BERTopic remained stable. Thus the paper highly recommends BERTopic for large and heterogenous datasets.

They managed to get around BERTs context limit of 512 tokens by chunking larger documents into sections, then generating an embedding for each chunk before weighing them together using mean pooling. The paper also points out that though not strictly necessary, BERTopic still significantly benefited from the preprocessed dataset aimed for the classical models.




== Summary <bg:sum>

To summarize, most research has shifted towards neural topic models as they more easily handle noisy data, and data of different languages. They have overall proven to be easier to use, not requiring the same amount of preprocessing as previous models which are more reliant on the @bow representation. 

Despite this, many researchers using topic modeling as a tool in their research still opt to use older models like LDA or NMF. This may be because even though @ntm:pl are more versatile, older probabilistic models still show competitive results on well behaved datasets @abdelrazek_eid_gawish_medhat_hassan_2023. Even on less well behaved datasets LDA have been shown to be usable by applying heavy lemmatization on the dataset @Bernadeta_2023.

This has led to the research on how well many @ntm:pl perform to be somewhat lacking. Even if it is not a complete research gap the field would benefit from more contributions into how these models perform different circumstances.

Even if the LDA can be made to generate useful results, they still often fall short of embedding based models. The topics generated by @ntm:pl such as BERTopic have been shown to be more distinct and more diverse, giving deeper insight into the data @abdelrazek_eid_gawish_medhat_hassan_2023 @egger

The biggest problem for @bow based models such as LDA have been shown to not be noise generated by for example OCR, but instead lemmatization @hall_mernitz_rensch_2026 @Hurtade_2024 @svensson2020exploring. OCR errors can often be handled with rather simple preprocessing such as stop-word removal or removing everything except common nouns. This has however proven insufficient in Swedish corpora as Swedish has much larger amount of morphological inflections and compound words than English texts @HEDLUND2001147.

This problem of lemmatization is easily circumvented by @ntm:pl as they use embeddings to assign similar meaning to all the different versions of words. The ability to take larger contexts into account and be pretrained on similar corpora lets the models interpret meaning from the text without getting confused by misspelled or inflected nouns.

The problem of handling historic corpora add another challenge to the mix. Though lemmatization tools had previously been sufficient to help the models like LDA to handle more difficult datasets, current lemmatization methods still struggle with handling historical and multilingual variances @hall_mernitz_rensch_2026. the performance particularly dips on more context reliant languages such as Ancient Greek.

Some have tried to handle historical contexts using manual methods such as "seeding". This means that they provide the model with words to which the topics should be formed around. This allows them to seed multiple variations of the same word into the same topic to help the model account for multiple words representing the same thing during different times in history. This does however require significant domain expertise and manual work which makes the method less applicable in many contexts @Hurtade_2024.

@ntm:pl does not always perform better than classical models on historical datasets@murugaraj-etal-2025-mining. Specifically BERTopic however has been shown to perform very well on historical data, outperforming other neural models and classical models the like in both topic coherence, diversity and scalability. 

When it comes to evaluation metrics, coherence is by far the most used metric, followed by diversity. Many studies however put emphasis on using human evaluation as well. As described in @bg:mtr, even though coherence and diversity are the gold standard metrics in topic modeling they are not without flaws and have been shown to produce misleading results @Hoyle. Despite this, some studies have argued that human evaluation is no better, with evaluators often give widely different scores to the same topics @hall_mernitz_rensch_2026.
