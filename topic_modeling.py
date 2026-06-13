# -*- coding: utf-8 -*-

#   pip install bertopic sentence-transformers umap-learn hdbscan
#   pip install gensim pyldavis nltk
import re
import pandas as pd
from IPython.display import display
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np, random, torch
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# =============================================================================
# STEP 0  Environment Check & Global Imports
# =============================================================================
if torch.cuda.is_available():
    print(f"GPU is enabled: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not detected.")

# =============================================================================
# STEP 0.5  Set Output Directory
#
# Purpose:
# - Define local directory for saving all model outputs, embeddings, and results
# - Directory is created automatically if it does not already exist
# =============================================================================
import os
SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# STEP 1  Import Data
#
# Data source: GitLab repository (4peerreview/icaconfpubs)
# - Loaded directly from raw URLs; no local download required
# - pandas.read_csv() accepts HTTP URLs natively
# =============================================================================
DATA_BASE = "data/processed"
papers   = pd.read_csv(f"{DATA_BASE}/papers.csv")
authors  = pd.read_csv(f"{DATA_BASE}/authors.csv")
sessions = pd.read_csv(f"{DATA_BASE}/sessions.csv")

# =============================================================================
# STEP 2.1  First-Round Cleaning: Remove Missing or Empty Abstracts
# =============================================================================
df = papers.dropna(subset=["Abstract"])
df = df[df["Abstract"].str.strip() != ""].reset_index(drop=True)
abstracts_raw = df["Abstract"].tolist()
years = df["Year"].tolist()
print(f"Total abstracts: {len(abstracts_raw)}, years {df['Year'].min()}–{df['Year'].max()}")

# =============================================================================
# STEP 2.2  Basic Text Cleaning
# =============================================================================
def clean_text(text: str) -> str:
    text = text.lower()                          # Convert to lowercase
    text = re.sub(r"http\S+", "", text)          # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)     # Remove numbers/punctuation, keep letters and spaces
    text = re.sub(r"\s+", " ", text).strip()     # Collapse multiple spaces
    return text

abstracts_clean = [clean_text(d) for d in abstracts_raw]

# =============================================================================
# STEP 2.3  Lemmatization
#
# Purpose:
# - Preserve original abstracts for embedding generation
# - Use lemmatized text for LDA and topic-word representation
# - Improve consistency across similar word forms
#
# Note:
# - BERTopic embeddings still use original abstracts
# - LDA uses lemmatized documents
# =============================================================================
import spacy
from tqdm.auto import tqdm

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # Disable parser and NER ── only lemmatization is needed for faster processing
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def lemmatize_docs(texts):
    docs_lemmatized = []

    for doc in tqdm(
        nlp.pipe(texts, batch_size=64),
        total=len(texts),
        desc="Lemmatizing abstracts"
    ):
        tokens = [
            t.lemma_
            for t in doc
            if not t.is_stop
            and len(t.text) > 1
            and not t.like_num
            and t.lemma_ != "-PRON-"
        ]
        docs_lemmatized.append(" ".join(tokens))

    return docs_lemmatized
print("Running lemmatization:")
abstracts_lemma = lemmatize_docs(abstracts_clean)
print(f"Lemmatization complete: {len(abstracts_lemma)} documents processed")

# =============================================================================
# STEP 2.4  Remove Empty Documents After Lemmatization
#
# Purpose:
# - Remove documents that become empty after preprocessing
# - Prevent empty texts from affecting topic modeling results
# - Keep original texts, processed texts, and metadata aligned
# =============================================================================
valid_idx = [i for i, d in enumerate(abstracts_lemma) if len(d.strip()) > 0]
n_removed = len(abstracts_lemma) - len(valid_idx)

print(f"Valid documents: {len(valid_idx)}, removed empty documents: {n_removed}")

abstracts_raw = [abstracts_raw[i] for i in valid_idx]     # list - Original texts for embeddings
abstracts_clean = [abstracts_clean[i] for i in valid_idx]
abstracts_lemma = [abstracts_lemma[i] for i in valid_idx] # list - Processed texts for LDA topic modeling
years = [years[i] for i in valid_idx]
df = df.iloc[valid_idx].reset_index(drop=True)            # csv - Metadata
df["abstract_raw"] = abstracts_raw
df["abstract_clean"] = abstracts_clean
df["abstract_lemmatized"] = abstracts_lemma

print(f"Alignment check: raw={len(abstracts_raw)}, clean={len(abstracts_clean)}, lemma={len(abstracts_lemma)}, df={len(df)}")

# =============================================================================
# STEP 2.5  Cache Lemmatized Documents
#
# Purpose:
# - Save preprocessing results to avoid repeated lemmatization
# - Store the filtered version after removing empty documents
# - Reload cached results after reconnecting to skip STEP 4–5
# =============================================================================
df.to_pickle(f"{SAVE_DIR}/processed_df.pkl")
print("Saved processed dataframe to Google Drive")

# Reload cached dataframe (skip STEP 4–5)
# df = pd.read_pickle(f"{SAVE_DIR}/processed_df.pkl")

# abstracts_raw = df["abstract_raw"].tolist()
# abstracts_clean = df["abstract_clean"].tolist()
# abstracts_lemma = df["abstract_lemmatized"].tolist()
# years = df["Year"].tolist()

# print(f"Loaded {len(df)} processed documents")

# =============================================================================
# STEP 3.1  Generate Sentence Embeddings
#
# Purpose:
# - Generate embeddings from original abstracts
# - Preserve semantic information for clustering
#
# Model options:
# all-MiniLM-L6-v2      -> faster, lightweight
# all-mpnet-base-v2     -> better quality, slower
# multilingual models   -> use for mixed-language corpora
#
# normalize_embeddings=True:
# normalize vectors for more stable similarity calculations
# =============================================================================
from sentence_transformers import SentenceTransformer

EMBED_NAME = "embeddings_mpnet_base.npy"
embed_model = SentenceTransformer("all-mpnet-base-v2")

print("Generating embeddings:")
embeddings = embed_model.encode(
    abstracts_raw,
    show_progress_bar=True,
    batch_size=128,
    normalize_embeddings=True
)
print("Embeddings shape:", embeddings.shape)   # Expected: (N, 768)

np.save(f"{SAVE_DIR}/{EMBED_NAME}", embeddings)
print("Saved embeddings to Google Drive")

# Reload cached embeddings
# embeddings = np.load(f"{SAVE_DIR}/{EMBED_NAME}")

# =============================================================================
# STEP 3.2  Configure UMAP for Dimensionality Reduction
#
# Parameters:
# n_neighbors  : Size of local neighborhood.
#                Smaller values (5–10) → more fine-grained topics
#                Larger values (30–50) → broader topics
#                Common starting point: 15
#
# n_components : Number of dimensions after reduction.
#                A value of 5 is commonly used for topic modeling
#
# min_dist     : Controls how tightly points are packed.
#                A value of 0.0 is commonly used for clustering tasks
#
# metric       : Distance metric used for similarity calculation.
#                Cosine distance is commonly used for text embeddings
#
# random_state : Fixed seed for reproducibility
# =============================================================================
from umap import UMAP

umap_model = UMAP(
    n_neighbors=20,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42)

# =============================================================================
# STEP 3.3  Configure HDBSCAN Clustering
#
# min_cluster_size :
#     Minimum number of documents required to form a topic.
#     Smaller values → more topics and finer granularity
#     Larger values → fewer and broader topics
#
# metric :
#     Distance metric used in reduced UMAP space.
#     Euclidean distance is commonly used after UMAP reduction
#
# cluster_selection_method :
#     "eom" generally produces fewer and more stable clusters
#
# prediction_data :
#     Enable prediction for new documents after training
#
# Note:
#     If min_samples is not specified,
#     HDBSCAN defaults it to min_cluster_size
# =============================================================================
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(
    min_cluster_size=200,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True)

# =============================================================================
# STEP 3.4  Configure CountVectorizer for c-TF-IDF Topic Representation
#
# Purpose:
# - Extract representative words for each topic using c-TF-IDF
# - Include both unigrams and bigrams for better phrase representation
#
# Parameters:
# min_df:
#     Minimum document frequency required to keep a term.
#     Higher values reduce noise and remove rare words.
#
# ngram_range:
#     (1, 2) allows both single words and meaningful bigrams
#     (e.g., "machine learning", "news media")
#
# Note:
# - Stopwords are already partially removed during lemmatization
# - Additional English stopword filtering can improve robustness
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(
    min_df=10,
    ngram_range=(1, 2),
    #stop_words="english"
)

# =============================================================================
# STEP 4  Train BERTopic Model
#
# top_n_words:
#     Number of words displayed per topic (does NOT affect clustering)
#
# nr_topics:
#     None   -> keep original number of topics
#     "auto" -> automatically merge similar topics
#     int    -> force a fixed number of topics
#
# calculate_probabilities:
#     If True, returns document-topic probability matrix (N × T)
#     Disabled here to reduce memory usage
#
# representation_model:
#     Defines how topic keywords are generated:
#     - keybert: semantic similarity-based keywords
#     - mmr: diversity-optimized keywords (reduces redundancy)
#     Note: these are alternative views, not merged outputs
# =============================================================================
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

representation_model = {
    "keybert": KeyBERTInspired(),
    "mmr": MaximalMarginalRelevance(diversity=0.3)}

topic_model = BERTopic(
    embedding_model=embed_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    top_n_words=15,
    nr_topics=None,
    calculate_probabilities=False,
    verbose=True)

print("Training BERTopic model...")
assert len(abstracts_raw) == len(abstracts_lemma)
assert embeddings.shape[0] == len(abstracts_lemma)
topics, probs = topic_model.fit_transform(abstracts_lemma, embeddings)

# Manual topic labeling (optional, recommended for publication)
#topic_model.set_topic_labels({0: "xxx", 1: "xxx", 2: "xxx"})

# Inspect topic representations
# Default: c-TF-IDF-based keywords
#topic_model.get_topic(0)

# Semantic representation (KeyBERT)
#topic_model.get_topic(0, representation="keybert")

# Diversity-optimized representation (MMR)
#topic_model.get_topic(0, representation="mmr")

# =============================================================================
# STEP 5  Topic Model Diagnostics
# Compute basic diagnostics to inspect topic structure, outlier proportion,
# and model granularity.
# =============================================================================
topic_info = topic_model.get_topic_info()

n_topics = len(topic_info[topic_info["Topic"] != -1])
n_noise = sum(t == -1 for t in topics)
noise_ratio = n_noise / len(topics)

print(f"\nGenerated topics: {n_topics}")
print(f"Outlier documents: {n_noise} ({noise_ratio:.1%})")
print("\nTop 10 topic summary:")
print(topic_info.head(10))

# =============================================================================
# STEP 6.1  Outliers Reduction
# =============================================================================
topics = topic_model.topics_
new_topics = topic_model.reduce_outliers(
    abstracts_lemma, topics,
    strategy="embeddings",
    embeddings=embeddings,
    threshold=0.3)

topic_model.update_topics(
    abstracts_lemma,
    topics=new_topics,
    vectorizer_model=vectorizer_model)

topics = new_topics
topic_info = topic_model.get_topic_info()

n_topic = len(topic_info[topic_info["Topic"] != -1])
n_out = sum(t == -1 for t in topics)
print(f"Final number of topics: {n_topic}, "
      f"Final outliers: {n_out} ({n_out/len(topics):.1%})")

# Topic Reduction has not been applied
# topic_model.reduce_topics(abstracts_lemma, nr_topics="auto")

# =============================================================================
# STEP 6.2  Topic Cluster Visualization
# =============================================================================
# Intertopic distance map
fig1 = topic_model.visualize_topics()
display(fig1)
# Topic keyword distributions
fig2 = topic_model.visualize_barchart(top_n_topics=15)
display(fig2)
# Hierarchical topic relationships
fig3 = topic_model.visualize_hierarchy()
display(fig3)
# Topic similarity heatmap
fig4 = topic_model.visualize_heatmap()
fig4

# =============================================================================
# STEP 7.1  Save Outputs
# =============================================================================
df_withresult = df.copy()
df_withresult["topic"] = topics
df_withresult["topic_prob"] = probs
topic_info.to_csv(
    f"{SAVE_DIR}/bertopic_topic_info.csv",
    index=False
    )
df_withresult.to_csv(
    f"{SAVE_DIR}/bertopic_paper_topics.csv",
    index=False
    )

fig1.write_html(f"{SAVE_DIR}/viz_topics.html")
fig2.write_html(f"{SAVE_DIR}/viz_barchart.html")
fig3.write_html(f"{SAVE_DIR}/viz_hierarchy.html")
fig4.write_html(f"{SAVE_DIR}/viz_heatmap.html")
print("Results saved successfully")

# =============================================================================
# STEP 7.2  Save Model
# =============================================================================
import shutil
import os

model_path = f"{SAVE_DIR}/bertopic_model"

if os.path.exists(model_path):
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    else:
        os.remove(model_path)

topic_model.save(
    model_path,
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model="all-mpnet-base-v2")
print("BERTopic model saved")

# Load model:
# topic_model = BERTopic.load(f"{SAVE_DIR}/bertopic_model")

# =============================================================================
# STEP 8.1  Merge Topics into 15 Categories
#
# Purpose:
# - Group fine-grained BERTopic topics into broader thematic categories
# - Reduce topic count to a more interpretable number for publication
#
# MERGE_MAPPING structure:
# - Keys   : descriptive category names (used as final topic labels)
# - Values : lists of original BERTopic topic IDs to merge
# - Topic IDs were assigned based on qualitative inspection of keywords
#   and domain knowledge of the communication field
#
# topics_to_merge:
# - Only groups with more than one topic ID are passed to avoid no-op merges
#
# After merge_topics():
# - BERTopic reassigns documents from merged IDs to the lowest ID in the group
# - update_topics() recomputes c-TF-IDF representations for merged categories
# =============================================================================
MERGE_MAPPING = {
    "Political Communication, Civic Engagement & Activism": [0, 15],
    "Gender, Culture & Identity":                           [1, 11, 17, 26],
    "Journalism & News Studies":                            [2, 6],
    "Video Game & Media Violence":                          [3],
    "PR, Crisis & Organizational Communication":            [4, 9, 21, 30],
    "Health Communication":                                 [5, 7, 24],
    "Advertising Effects & Online Consumer Behavior":       [10],
    "Interpersonal, Relational & Parasocial Communication": [12, 18, 29],
    "Environment & Science Communication":                  [14],
    "Intercultural & Global Communication":                 [16, 20, 31],
    "Children, Family Media Use & Parental Mediation":      [23],
    "Communication Technology & Digital Media":             [8, 13, 19],
    "Media Policy, Platform Governance & Privacy":          [22, 27, 28],
    "Sports Media":                                         [33],
    "Communication Theory & Disciplinary Reflection":       [32],
}

topics_to_merge = [
    topic_ids
    for topic_ids in MERGE_MAPPING.values()
    if len(topic_ids) > 1
]

topic_model.merge_topics(
    abstracts_lemma,
    topics_to_merge
)

topic_model.update_topics(
    abstracts_lemma,
    vectorizer_model=vectorizer_model
)

print(topic_model.get_topic_info()[["Topic", "Count", "Name"]].to_string())

# =============================================================================
# STEP 8.2  Assign Category Labels to Merged Topics
#
# Purpose:
# - Replace numeric topic IDs with descriptive category names for display
# - Labels appear in BERTopic visualizations and exported CSV outputs
#
# representative_id:
# - After merging, BERTopic consolidates a group under the lowest topic ID
# - min(topic_ids) selects that canonical ID as the key for labeling
#
# set_topic_labels():
# - Maps each representative topic ID to its human-readable category name
# - Labels are shown in all visualizations produced by topic_model.visualize_*()
# =============================================================================
topic_name_mapping = {}
for category_name, topic_ids in MERGE_MAPPING.items():
    representative_id = min(topic_ids)
    topic_name_mapping[representative_id] = category_name

topic_model.set_topic_labels(topic_name_mapping)

# =============================================================================
# STEP 9.1  Export Initial Category Keywords
#
# Purpose:
# - Extract the top 15 c-TF-IDF keywords for each merged category
# - Save as a CSV before keyword refinement to enable before/after comparison
#
# Note:
# - Representations here still use the original vectorizer from STEP 3.4
# - representative_id = min(topic_ids) accesses the merged topic's keyword list
# - This snapshot is useful for diagnosing which terms get removed in STEP 9.2–9.3
# =============================================================================
merged_keywords = []
for category_name, topic_ids in MERGE_MAPPING.items():
    representative_id = min(topic_ids)
    words_scores = topic_model.get_topic(representative_id)
    if words_scores:
        keywords = [word for word, score in words_scores[:15]]
    else:
        keywords = []
    merged_keywords.append({
        "category": category_name,
        "representative_topic_id": representative_id,
        "keywords": ", ".join(keywords)
    })
df_merged_keywords = pd.DataFrame(merged_keywords)

print(df_merged_keywords.to_string())

df_merged_keywords.to_csv(
    f"{SAVE_DIR}/bertopic_merged_keywords.csv",
    index=False
)

# =============================================================================
# STEP 9.2  Recompute Topic Representations
#
# Recalculate topic representations using c-TF-IDF with a customized
# stopword list to improve keyword interpretability.
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

english_stopwords = set(stopwords.words("english"))

domain_stopwords = {
    "study", "paper", "research",
    "result", "results",
    "finding", "findings",
    "use", "used", "using",
    "also", "however", "thus",
    "therefore",
    "data", "analysis",
    "based", "approach",
    "propose", "proposed",
    "examine", "suggest",
    "one", "two", "three",
    "first", "second", "third",
    "among", "may", "show",
    "present", "communication",
    "datum", "model",
    "process", "develop",
    "provide", "base",
    "concept", "theory",
    "studies", "participants"
}

all_stopwords = list(english_stopwords | domain_stopwords)

vectorizer_clean = CountVectorizer(
    min_df=10,
    ngram_range=(1, 2),
    stop_words=all_stopwords
)

topic_model.update_topics(
    abstracts_lemma,
    vectorizer_model=vectorizer_clean,
    top_n_words=50
)

print("Topic representations updated using c-TF-IDF.")

# =============================================================================
# STEP 9.3  Lemma-Based Keyword Deduplication
#
# Remove redundant keywords that share the same lemma while preserving
# the highest-ranked occurrence.
# =============================================================================
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_topic_representations(topic_model, nlp, topic_name_mapping, keep_n=15):
    cleaned_representations = {}
    drop_logs = []

    for topic_id in topic_name_mapping.keys():
        words_scores = topic_model.get_topic(topic_id)
        if not words_scores:
            continue

        words  = [w for w, s in words_scores]
        scores = [s for w, s in words_scores]
        lemma_map = {w: " ".join(token.lemma_ for token in nlp(w)) for w in words}

        seen_lemmas = {}
        keep_indices = []
        for i, word in enumerate(words):
            lemma = lemma_map[word]
            if lemma not in seen_lemmas:
                seen_lemmas[lemma] = i
                keep_indices.append(i)
            else:
                drop_logs.append({
                    "topic_id":     topic_id,
                    "category":     topic_name_mapping[topic_id],
                    "dropped_word": word,
                    "lemma":        lemma,
                    "duplicate_of": words[seen_lemmas[lemma]]
                })

        cleaned_representations[topic_id] = [
            (words[i], scores[i]) for i in keep_indices[:keep_n]
        ]

    return cleaned_representations, pd.DataFrame(drop_logs)


cleaned_representations, df_drop_log = clean_topic_representations(
    topic_model, nlp, topic_name_mapping, keep_n=15
)

for topic_id, words_scores in cleaned_representations.items():
    topic_model.topic_representations_[topic_id] = words_scores

print(f"Removed {len(df_drop_log)} duplicated keywords.")

# =============================================================================
# STEP 9.4  Export Final Keywords
#
# Save cleaned keywords after c-TF-IDF recalculation and lemma-based
# deduplication.
# =============================================================================
final_keywords = []
for topic_id, category_name in topic_name_mapping.items():
    words_scores = cleaned_representations.get(topic_id, [])
    keywords = [word for word, score in words_scores[:15]]
    final_keywords.append({
        "topic_id": topic_id,
        "category": category_name,
        "keywords": ", ".join(keywords)
    })
df_final_keywords = pd.DataFrame(final_keywords)

print(df_final_keywords.to_string())

# =============================================================================
# STEP 9.5  Save All Final Outputs
#
# Purpose:
# - Save keyword table and deduplication log from STEP 9.3–9.4
# - Attach final (post-merge) topic IDs to the paper-level dataframe
# - Save as a unified CSV for use in the analysis phase (STEP 10.2 onwards)
#
# topic_model.topics_:
# - Contains the current topic assignment for each document
# - Updated after merge_topics() and reduce_outliers() (STEP 6–8)
# - Outlier documents retain the assignment value -1
#
# Output files:
# - bertopic_final_keywords.csv        : per-category keywords after deduplication
# - bertopic_keyword_dedup_log.csv     : log of dropped duplicate keywords
# - bertopic_paper_topics_final_15.csv : canonical paper-topic table for STEP 10.2
# =============================================================================
df_final_keywords.to_csv(f"{SAVE_DIR}/bertopic_final_keywords.csv", index=False)
df_drop_log.to_csv(f"{SAVE_DIR}/bertopic_keyword_dedup_log.csv", index=False)
print("Final keywords exported.")

df["topic_id"] = topic_model.topics_
df.to_csv(f"{SAVE_DIR}/bertopic_paper_topics_final_15.csv", index=False)
print(f"Saved {len(df)} papers with topic assignments")
print(df["topic_id"].value_counts().sort_index())

# =============================================================================
# STEP 10.1  Reload Analysis Dependencies
#
# Purpose:
# - Re-import all required libraries for the analysis phase
# - Set the output directory for analysis figures and CSVs
#
# Note:
# - This block is designed to run independently from the topic modeling steps
# - SAVE_DIR should point to the directory where STEP 7–9 outputs were saved
# =============================================================================
import os
import re
import numpy as np
import pandas as pd
import torch
import spacy
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# STEP 10.2  Load Finalized Topic Assignment Data
#
# Purpose:
# - Load the CSV containing per-paper topic assignments produced in STEP 8–9
# - Each row is one paper; relevant columns include "topic_id" and "Year"
# =============================================================================
df_bertopic = pd.read_csv(f"{SAVE_DIR}/bertopic_paper_topics_final_15.csv")

# =============================================================================
# STEP 10.3  Define Topic Mapping and Short Display Labels
#
# Purpose:
# - Map numeric topic IDs (0–14) to full descriptive category names
# - Define shorter display labels for axis readability in exported figures
# - Filter out outlier documents (topic_id == -1) before analysis
#
# Note:
# - topic_name_map keys correspond to re-indexed IDs after merging (STEP 8.1–8.2)
# - short_labels must preserve the same order as topic_name_map (ID 0–14)
# =============================================================================
topic_name_map = {
    0:  "Political Communication, Civic Engagement & Activism",
    1:  "Gender, Culture & Identity",
    2:  "Communication Technology & Digital Media",
    3:  "Health Communication",
    4:  "Journalism & News Studies",
    5:  "PR, Crisis & Organizational Communication",
    6:  "Interpersonal, Relational & Parasocial Communication",
    7:  "Intercultural & Global Communication",
    8:  "Media Policy, Platform Governance & Privacy",
    9:  "Video Game & Media Violence",
    10: "Advertising Effects & Online Consumer Behavior",
    11: "Environment & Science Communication",
    12: "Children, Family Media Use & Parental Mediation",
    13: "Communication Theory & Disciplinary Reflection",
    14: "Sports Media",
}

short_labels = [
    "Political Comm",
    "Gender & Culture",
    "Comm Technology",
    "Health Comm",
    "Journalism",
    "PR & Org Comm",
    "Interpersonal Comm",
    "Intercultural Comm",
    "Media Policy",
    "Video Games",
    "Advertising",
    "Environment Comm",
    "Children & Family",
    "Comm Theory",
    "Sports Media",
]

df_plot = df_bertopic[df_bertopic["topic_id"] != -1].copy()   # Exclude outlier documents
df_plot["topic_name"] = df_plot["topic_id"].map(topic_name_map)

# =============================================================================
# STEP 10.4  Compute Topic Frequency and Annual Proportional Prevalence
#
# Purpose:
# - count_pivot : raw paper counts per (Year × topic) cell
# - prop_pivot  : row-normalized proportions (each year sums to 1.0)
# - topic_counts: total paper count per topic across all years
#
# Note:
# - Columns are reordered to match topic IDs 0–14 for consistent color mapping
# =============================================================================
count_pivot = df_plot.groupby(["Year", "topic_name"]).size().unstack(fill_value=0)
prop_pivot  = count_pivot.div(count_pivot.sum(axis=1), axis=0)

topic_counts = df_plot.groupby("topic_name").size()
topic_counts = topic_counts[[topic_name_map[i] for i in range(15)]]   # Reorder to match topic ID sequence
prop_pivot   = prop_pivot[[topic_name_map[i] for i in range(15)]]     # Reorder to match topic ID sequence

# =============================================================================
# STEP 10.5  Define Color Palette
#
# Purpose:
# - Assign one consistent color per topic across all figures
# - Use matplotlib's tab20 palette for 15 visually distinct colors
#
# Note:
# - tab20 provides 20 colors; only the first 15 are used here
# - Color index matches topic ID (0–14) across all plots
# =============================================================================
COLORS = list(plt.cm.tab20.colors[:15])

# =============================================================================
# STEP 11.1  Two-Panel Visualization: Topic Distribution and Temporal Trends
#
# Figure layout:
# (A) Horizontal bar chart — overall topic frequency across the full corpus
# (B) Line chart — annual proportional prevalence per topic (2003–2018)
#
# Parameters (bar chart):
# barh              : horizontal bars, sorted by count ascending so the
#                     largest bar appears at the top
# edgecolor="white" : thin white separator between bars for readability
#
# Parameters (line chart):
# rolling window=3  : 3-year centered rolling mean (precomputed for optional use;
#                     raw proportions are plotted in this version)
# marker="o"        : circle marker at each annual observation point
# markersize=3      : small markers to avoid visual clutter at 16 years
# alpha=0.85        : slight transparency to reveal overlapping trend lines
# bbox_to_anchor    : legend placed outside the right edge of the chart
# MultipleLocator(2): x-axis tick every 2 years to prevent label crowding
#
# Output:
# - PDF  : vector format for publication submission
# - PNG  : 300 dpi raster for slides and previews
# =============================================================================
fig, (ax_bar_h, ax_line) = plt.subplots(1, 2, figsize=(18, 7))

# --- (A) Horizontal Bar Chart: Overall Topic Frequency ---
sorted_counts = topic_counts.sort_values()   # Ascending so largest bar appears at top
sorted_labels = [short_labels[list(topic_counts.index).index(name)] for name in sorted_counts.index]
sorted_colors = [COLORS[list(topic_counts.index).index(name)] for name in sorted_counts.index]

ax_bar_h.barh(
    sorted_labels,
    sorted_counts.values,
    color=sorted_colors,
    edgecolor="white",
    linewidth=0.5,
)
ax_bar_h.set_xlabel("Number of Papers")
ax_bar_h.set_title("(A) Overall Topic Distribution", fontsize=12, fontweight="bold")
ax_bar_h.spines["top"].set_visible(False)
ax_bar_h.spines["right"].set_visible(False)

# --- (B) Line Chart: Annual Proportional Prevalence ---
prop_smooth = prop_pivot.rolling(window=3, center=True).mean()   # Precomputed; available for smoothed version
for i, col in enumerate(prop_pivot.columns):
    ax_line.plot(
        prop_pivot.index,
        prop_pivot[col],
        color=COLORS[i],
        linewidth=1.8,
        marker="o",
        markersize=3,
        alpha=0.85,
        label=short_labels[i],
    )
ax_line.set_xlabel("Year")
ax_line.set_ylabel("Proportion of Annual Papers")
ax_line.set_title("(B) Topic Proportional Prevalence Over Time",
                   fontsize=12, fontweight="bold")
ax_line.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7.5, frameon=False)
ax_line.xaxis.set_major_locator(mticker.MultipleLocator(2))   # Tick every 2 years
ax_line.spines["top"].set_visible(False)
ax_line.spines["right"].set_visible(False)

fig.suptitle("BERTopic Analysis — ICA Corpus 2003–2018", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig_bar_line_bertopic.pdf", bbox_inches="tight")
plt.savefig(f"{SAVE_DIR}/fig_bar_line_bertopic.png", bbox_inches="tight", dpi=300)
plt.show()
