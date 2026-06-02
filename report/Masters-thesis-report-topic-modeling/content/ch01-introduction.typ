= Introduction <intro>


== background


Riksarkivet houses Sweden's largest historical archive of documents, some of which dating back more than a thousand years. It is part of Riksarkivets mission to store the historical data, but also to make it available for the public@det_här_är_riksarkivet_2026 @Förordning.

To navigate these massive data collections, efficient information retrieval is required. Relying on simple keyword search is often insufficient. Since the information should be usefull to humans, the retrieval process must be easy to understand and use. Topic modeling is a powerful tool that can help structuring and organizing the large quantities of textual documents in an interpretable way@abdelrazek_eid_gawish_medhat_hassan_2023. This is achieved by assigning topics to documents based on their content. There are however many different techniques within topic modeling, all with their own advantages and disadvantages. The dataset at Riksarkivet contains many challenging properties. Most research has been focused on modern English texts@vayansky_kumar_2020, but the data in this case is both in Swedish and will contain historical variance as words change or get replaced over time. Because a large amounts of the documents have been stored physically, they have been transcribed using OCR @mori. This has caused the dataset to contain a large amount of noise. tackling this dataset is a unique challenge and will require careful testing on what topic modeling technique produce the best results.


== Problem <intr:prbl>

The research question this thesis aims to answer is: 

_What topic modeling techniques produce the best results on the dataset at Riksarkivet? _

The technique most appropriate for the task must be able to handle noisy data caused by OCR transcriptions. In this context, noisy means that some words may be misspelled or sometimes complete nonsense. The techniques must also be able to handle historical variance since many words change meaning or spelling over time. 

This study will investigate and compare four different modeling techniques on the specific dataset. The models that will be tested are: Latent Dirichlet Allocation (LDA)@blei_edu_ng_jordan_edu_2003, Embedded Topic Model (ETM) @dieng_ruiz_blei_2020, Contextualized Topic Model (CTM)@bianchi_terragni_hovy_nozza_fersini_2021 and BERTopic @Maarten_G_2022. CTM and BERTopic are state of the art @ntm:pl, ETM is one of the earlier @ntm:pl and LDA is an older probabilistic model that will be used as a baseline.

The performance of these models will mainly be measured and compared primarily using two metrics: topic coherence and topic diversity. Topic coherence is a measure of how well the words in a topic fit together. Topic diversity is a measure of how distinct the topics are. Since the purpose is to yield a result that is interpretable by humans, there will also be an element of human evaluation.

Even though LDA and ETM are older models it is still of interest to investigate their performance. Since the dataset is ill behaved it is important to check whether the improvements of the state of the art models is reflected in the results or if the classic approaches are sufficient. CTM and BERTopic are both based on BERT and may need fine tuning in order to interpret the documents in the dataset. This may result in models of different quality compared to other studies. For this reason it is important to include models that lack the ability to interpret semantic meaning as comparisons.

=== Hypothesis
The abilities of BERT to take context into account will help mitigate the problems caused by language changing throughout history. This will make BERTopic and CTM perform better than the alternatives. While ETM is expected to outperform LDA due to its handling of rare words, the BERT-based models (CTM and BERTopic) are hypothesized to achieve the highest coherence by leveraging semantic context.

These results will be shown by a larger topic coherence and topic diversity score by the better performing models. However, the results will not be overwhelmingly in favor for BERTopic and CTM as it has been shown that neural models generally perform relatively worse on the standard metrics despite showing better results when evaluated by humans.


== Purpose

the purpose of this thesis is to provide insight into what Topic modeling techniques are best suited for the dataset of Riksarkivet. By investigating how noisy data with great historical variance impacts the effectiveness of different techniques, Riksarkivet will receive guidance on how to keep their infrastructure efficient for both professional and general use.

Riksarkivet is an important entity for preserving documents for future generations and also make that information available digitally. As the amount of data that has to be stored at the archive has dramatically increased in the last decades, the need to organize it has subsequently also increased. Riksarkivet must be able to make efficient retrieval possible for both research and general usage. guaranteeing this service is part of their mission received by the Swedish government@Förordning.

This thesis is part of a larger project where Riksarkivet aims to improve their information retrieval capabilities in their database. Exploring the usefulness of sorting the documents into relevant topics with topic modeling is a key part of this.

Scientifically the thesis will contribute to further evaluating the models on different kinds of datasets. Mostly models are just evaluated on well behaved datasets; this study aims to expand that to a more problematics one. Even though noise is a common problem in datasets, multilingual properties and especially Swedish data has less precedence in the literature. Since the data comes from a wide timeframe this will provide a somewhat novel environment to test the performance of the different models.



== Goals

The goal of this thesis is to through a comparative study provide an answer for what topic modeling technique is best suited for the OCR-transcribed historical documents at Riksarkivet. This will be achieved through the following subgoals:

#enum(indent:3em,spacing: 1em, 
[
  Set up a pipeline for running the different topic models on the dataset.
],
[
  Preprocess the data to accommodate preferred format of the models as much as possible.

],
[
  Evaluate the models performance using coherency and diversity measures and comparatively judge their performance.
],
[
  Provide clear guidance on what models are most appropriate to apply on the data at Riksarkivet or if topic modeling is a worthwhile approach at all.
])

== Methodology


A large part of the methodology will involve handling the dataset. Many models rely on the @bow representation either entirely or partially to create the topics. This means that the data must be stored in both raw text form and as word frequencies. Additionally the @bow representation is often impeded by stop-words, even the models that can handle stop words will benefit from their removal. Therefore stop-words must be removed from the @bow representations of the dataset. 

Once the dataset has been sufficiently pre-processed we must fine tune the a BERT model. Ideally, a BERT model pre-trained on Swedish would be best. The historic Swedish data infers a significant domain shift from what the BERT models are trained on and must @haffenden2023making @malmsten2020playingwordsnationallibrary. Luckily, the Swedish royal library has trained such a model, known as KB-BERT @malmsten2020playingwordsnationallibrary. The dataset in this thesis is still however a domain shift from what KB-BERT was trained on. The BERT model must therefore be fine-tuned before it can be utilized for both the CTM and BERTopic, as is standard practice in @nlp @gardazi_daud_malik_bukhari_tariq-alsahfi_bader_alshemaimri_2025. The fine-tune will be performed in an unsupervised manner on the same corpus as the topic modeling will be performed on using @mlm objective @devlin_chang_lee_toutanova_2019 @taylor_1953.

Once the model is fine-tuned it can be applied to the relevant models. The different models will then be applied to the dataset and produce their topics. The generated topics can then be evaluated based on their most common words. This study will follow the common practice and evaluate the models mainly using intrinsic measures, though limited they are widely used and allows for comparisons with previous studies @abdelrazek_eid_gawish_medhat_hassan_2023. Specifically, topic coherence and topic diversity are the metrics that will be used. 

Topic coherence will investigate the most popular words in a topic and describe how well they "fit together". Topic diversity will instead look at how unique the topics are. High diversity means that the topics have very few words in common.

To run the evaluations the @octis framework will be used. @octis is a framework aimed at training, analyzing and comparing topic models and has been used in multiple previous studies.

If possible, human evaluation will also be used to a smaller extent.
== Deliminations

This section describes how the scope of the thesis has been limited due to time, storage or compute restraints.

#list(indent: 2em,
[
  *Evaluation: * No large scale human evaluation will be conducted. Instead the intrinsict measures of coherence and diversity will be used. This makes the evaluation more simple to conduct and allows for comparisons with previous studies as those measures are widely used in the topic modeling domain.

],[
  *Information retrieval: * This paper will strictly focus on performance of topic modeling on the dataset. Even though the final goal is to later use the topics for information retrieval, this step is not investigated in this paper. 
  
],[
  *Models: * Due to the large amount of different topic models, only few of them have been considered in this study, the majority of which are considered @ntm:pl as that is state of the art.
])



== Structure of the Thesis

The thesis is divided up into the following sections:

@bg describes the relevant background to understand the method and model decisions. It describes all the relevant topic models as well as some older ones in @bg:TM. @bg:mtr then goes on to introduce the evaluation metrics that will be used in the theses. It explains how the metrics work and what their strengths and weaknesses are, as well as what alternatives exist.

@bg:bert and @bg:sbert go more in depth into the underlying @nlp methods that work as a base for some of the context based models and how they are applied to them.

The concept of OCR errors are very relevant for this thesis and are explain more in @bg:ocr.

Lastly it also contains multiple subsections for related work in @bg:rw. This includes studies that simply compare the models to each other as well as studies that evaluates the models on similarly troublesome datasets.

@mth contains the main methodology. It gives a description of all the steps of the methodology in @mth:rp. It also describes the dataset in more detail and how its problematic nature will be handled in terms of preprocessing in @mth:data. This chapter contains @mth:tm that describes the actual models and specifically how they will be applied to the dataset. The same section also explains how they are set up and in the case of contextual models, how they are fine tuned. @mth:exp finally goes into detail into what specific software and libraries are used. It also explains the hyperparameters and the evaluation metrics.

@res details all the results yielded after running the experiments. It is split into two main sections. First @res:ind that shows the detailed result and values recorded for the metrics for all the models and their variations individually. Then @res:sum focuses on the most successful versions of all the models and compare them to each other to give indication of which one was the most effective at handling the specific dataset.

While the results section display all the results, @disc focuses on analyzing the results and what they mean in regards to hypothesis and research question. This chapter also attempts to view the results in a wider context to see what other conclusions can be drawn.

@conc summarizes the most important conclusions from @disc in @conc:conc and reviews what limitations existed in @conc:limitations. Finally @conc:fw discusses what areas could have been further investigated in this study. This is done both in the context of yielding better results but also investigating the area further based on the results yielded.


