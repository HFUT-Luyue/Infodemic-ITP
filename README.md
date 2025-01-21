# Infodemic-ITP

## Introduction
This repository helds the code utilized in the main experiment of "Responsible AI-Enabled Infodemic Management: A Hypergraph-based Infodemic Topic Prediction Framework."

## Desciption of this repository

### Dataset

We utilize two public dataset: Coronavirus tweets dataset (https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset)  by Lamsal 2020 and "A Monkeypox Twitter Dataset" (https://data.mendeley.com/datasets/242whtdt3m/1) by Nia et al. 2023. Please download the dataset and complete the necessary hydration process.

### Code

We provide a number of .ipynb files for reader friendliness and ease for access and debug. We removed the dataset file path for privacy reasons, and user can choose their own file path.

[runtimex.py] : This file is required to run some of the ipynb files, and contains several methods that can be utilized for multiprocessing.

[Bulk Separation.ipynb] : If the hydration result is a huge jsonl file per day, we use this to separate the post corpus into 100 posts per file, putting them in their according date.

[DTM-processing.ipynb] : Once done, Dynamic Topic Modeling implemented by Blei & Lafferty (2006) Tomotopy (Lee 2023) could be utilized to generate topics for the chosen keywords and categorize each post into topics.

[High-level Feature Extract.ipynb] : Once the topics and posts are in position, we extract high-level features such as user generated profile subjectivity, ARI, etc., from posts in a topic.

[Low-level Feature Extract & Keyword Processing.ipynb] : We also extract the embedding for each topic using BERT-based models. A unique keyword list is created in this file, then top-10 keywords for each topic timepoint is recorded in file for easier dataloading.

[Labeling with Poynter Dataset.ipynb] : We perform the labeling process mentioned in our paper.

[ITP-Main.ipynb] : With topic features, their top keywords (used for hyperedge building), and their label available, we can use these to test the effectiveness of our framework. We left the latest run of our best implementation in for result verification.


## References
Blei DM, Lafferty JD (2006) Dynamic topic models. Proceedings of the 23rd international conference on Machine learning. ICML ’06. (Association for Computing Machinery, New York, NY, USA), 113–120.

Lamsal R (2020) Coronavirus (COVID-19) Tweets Dataset.

Nia ZM, Bragazzi NL, Wu J, Kong JD (2023) A Twitter dataset for Monkeypox, May 2022. Data in Brief 48:109118.

Lee M (2023) tomotopy.
