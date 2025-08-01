#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Clustering Algorithm
A complete implementation for clustering text data using sentence embeddings
"""
from agent import Agent
from typing import Literal
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import os
from math import log2
from embedding.embed import Embedding, reduce_embeddings
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from collections import OrderedDict
import pandas as pd
import warnings
import time
import re
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from agent import AsyncMultiAgent
import json
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
import asyncio

warnings.filterwarnings("ignore")


class TextClusteringConfig:
    """Configuration class for clustering parameters"""

    def __init__(self,
                 output_path="./output",
                 provider="sentence_transformers",
                 embedding_model="Qwen/Qwen3-Embedding-0.6B",
                 embedding_reduce_method: Literal["pca",
                                                  "tsne", "umap"] = "umap",
                 sim_threshold=0.70,
                 group_sim_cutoff=0.45,
                 min_cluster_size=3,
                 max_data_points=560000,
                 chunk_size=15000,
                 label_with: Literal['topics', 'summary'] = 'summary',
                 label_agent: str = "gpt-4.1",
                 label_provider: str = "openai"):
        self.output_path = Path(output_path)
        self.sim_threshold = sim_threshold
        self.group_sim_cutoff = group_sim_cutoff
        self.min_cluster_size = min_cluster_size
        self.max_data_points = max_data_points
        self.chunk_size = chunk_size
        self.output_path.mkdir(exist_ok=True)
        self.provider = provider
        self.embedding_model = embedding_model
        self.embedding_reduce_method = embedding_reduce_method
        self.label_with = label_with
        self.label_agent = label_agent
        self.label_provider = label_provider


class ClusterLabeler:
    """Enhanced cluster analysis with keyword extraction and intelligent labeling"""

    def __init__(self, config: TextClusteringConfig):
        """Initialize the cluster labeler with necessary components"""
        try:
            self.config = config
            self.label_agent = Agent(
                name="label_agent",
                prompt="",
                description="An agent that generates labels for clusters based on the topics discussed in the conversation.",
                model=config.label_agent,
                provider=config.label_provider,
            )
            self.async_label_agent = AsyncMultiAgent(
                name="label_agent",
                prompt="",
                description="An agent that generates labels for clusters based on the topics discussed in the conversation.",
                model=config.label_agent,
                provider=config.label_provider,

            )
            # Initialize lemmatizer and stopwords
            self.lemmatizer = WordNetLemmatizer()
            self.en_stopwords = stopwords.words(
                'english') + ['hi', 'know', 'u', 'n', 'w', 'h']

            # Remove specific words from stopwords
            for word in ['out', 'you', 'there', 'not', 'now', 'no']:
                if word in self.en_stopwords:
                    self.en_stopwords.remove(word)

            # Load spaCy model
            self.nlp = spacy.load('en_core_web_sm', disable=['ner'])

        except Exception as e:
            print(f"Warning: Some components failed to initialize: {e}")
            self.lemmatizer = None
            self.en_stopwords = []
            self.nlp = None

    def preprocess_text(self, text):
        """Preprocess text using spaCy for lemmatization and cleaning"""
        if not self.nlp:
            return text.lower()

        text = text.lower()
        doc = self.nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        text = ' '.join(lemmatized_tokens)

        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha()]

        # Remove stopwords
        stopwords_set = set(self.en_stopwords)
        tokens = [word for word in tokens if word not in stopwords_set]

        return ' '.join(tokens)

    def prepare_cluster_data(self, df, col_name):
        """Prepare cluster data for analysis"""
        # Preprocess text
        tqdm.pandas()

        df['processed_text'] = df['summary'].progress_apply(
            self.preprocess_text)

        def concatenate_insights(cluster):
            insights = cluster['processed_text']
            return ' '.join(insights)

        def count_insights(cluster):
            return len(cluster['processed_text'].tolist())
        grouped_insights = df.groupby(
            col_name).progress_apply(concatenate_insights)

        clustered_insights = grouped_insights.to_dict()

        return clustered_insights

    def get_top_n_words(self, text, n=10):
        """Get top N words from text"""
        tokens = nltk.word_tokenize(text)
        freq_dist = nltk.FreqDist(tokens)
        sorted_freq = sorted(
            freq_dist.items(), key=lambda x: x[1], reverse=True)
        sorted_freq = [((word,), freq) for word, freq in sorted_freq]
        return sorted_freq[:n]

    def get_top_n_bigrams(self, text, n=10):
        """Get top N bigrams from text"""
        tokens = nltk.word_tokenize(text)
        bigrams = nltk.bigrams(tokens)
        freq_dist = nltk.FreqDist(bigrams)
        sorted_freq = sorted(
            freq_dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_freq[:n]

    def get_top_n_trigrams(self, text, n=10):
        """Get top N trigrams from text"""
        tokens = nltk.word_tokenize(text)
        trigrams = nltk.trigrams(tokens)
        freq_dist = nltk.FreqDist(trigrams)
        sorted_freq = sorted(
            freq_dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_freq[:n]

    def safe_log2(self, x):
        """Safe logarithm calculation"""
        return log2(x) if x > 0 else 0

    def compute_information_gain(self, freq_df):
        """Compute information gain scores for terms across clusters"""
        ig_scores = {}
        term_totals = freq_df.sum(axis=0)
        cluster_totals = freq_df.sum(axis=1)
        grand_total = term_totals.sum()

        for term in freq_df.columns:
            ig_scores[term] = {}
            for cluster in freq_df.index:
                A = freq_df.loc[cluster, term]
                B = term_totals[term] - A
                C = cluster_totals[cluster] - A
                D = grand_total - (A + B + C)

                pt = (A + B) / grand_total
                pnot_t = (C + D) / grand_total
                pc = (A + C) / grand_total
                pnot_c = (B + D) / grand_total

                ig = 0
                for count, px, py in [(A, pt, pc), (B, pt, pnot_c), (C, pnot_t, pc), (D, pnot_t, pnot_c)]:
                    pxy = count / grand_total
                    if pxy > 0:
                        ig += pxy * self.safe_log2(pxy / (px * py))

                ig_scores[term][cluster] = ig

        return pd.DataFrame(ig_scores).T

    def merge_phrases_simple(self, phrase_data):
        """Merge phrases and return unique sequences, prioritizing longer phrases"""
        sorted_phrases = sorted(phrase_data, key=lambda x: (
            len(x[0]), x[1]), reverse=True)

        unique_sequences = set()
        result_phrases = []

        # First pass: collect all unique word sequences
        for phrase, freq in sorted_phrases:
            phrase_str = ' '.join(phrase)
            unique_sequences.add(phrase_str)

        # Second pass: build merged phrases, avoiding subsequences
        used_words_sets = []

        for phrase, freq in sorted_phrases:
            current_words = set(phrase)
            phrase_str = ' '.join(phrase)

            # Check if this phrase is a subsequence of any already processed phrase
            is_subsequence = False
            for existing_phrase in result_phrases:
                if phrase_str in existing_phrase and phrase_str != existing_phrase:
                    is_subsequence = True
                    break

            if not is_subsequence:
                # Try to extend this phrase
                extended_phrase = list(phrase)
                used_phrases = {phrase}

                # Keep extending while possible
                extended = True
                while extended:
                    extended = False
                    last_word = extended_phrase[-1]

                    # Look for phrases that start with the last word
                    for other_phrase, other_freq in sorted_phrases:
                        if other_phrase in used_phrases:
                            continue

                        if len(other_phrase) > 0 and other_phrase[0] == last_word:
                            # Check if adding this would create a subsequence of existing result
                            potential_extension = extended_phrase + \
                                list(other_phrase[1:])
                            potential_str = ' '.join(potential_extension)

                            # Only extend if it doesn't create a subsequence
                            would_be_subsequence = False
                            for existing in result_phrases:
                                if potential_str in existing and potential_str != existing:
                                    would_be_subsequence = True
                                    break

                            if not would_be_subsequence:
                                extended_phrase.extend(other_phrase[1:])
                                used_phrases.add(other_phrase)
                                extended = True
                                break

                final_phrase = ' '.join(extended_phrase)
                if final_phrase not in result_phrases:
                    result_phrases.append(final_phrase)

        return result_phrases

    def extract_cluster_phrases(self, clustered_insights):
        """Extract and merge phrases for each cluster using multiple approaches"""
        # Create frequency matrix
        vectorizer = CountVectorizer(ngram_range=(1, 3))
        X = vectorizer.fit_transform(clustered_insights.values())
        terms = vectorizer.get_feature_names_out()
        freq_df = pd.DataFrame(
            X.toarray(), index=clustered_insights.keys(), columns=terms)

        # Compute information gain
        ig_df = self.compute_information_gain(freq_df)

        # Extract phrases for each cluster
        cluster_phrases = {
            'merged_phrases_all': {},
            'merged_phrases_ig': {},
            'merged_phrases_ngrams': {},
            'top_words': {},
            'top_bigrams': {},
            'top_trigrams': {}
        }

        for cluster_number, insights in clustered_insights.items():
            # Information gain based phrases
            words = ig_df[cluster_number].nlargest(30).index.tolist()
            scores = ig_df[cluster_number].nlargest(30).values.tolist()

            # N-gram based phrases
            top_words = self.get_top_n_words(insights, n=15)
            top_bigrams = self.get_top_n_bigrams(insights, n=15)
            top_trigrams = self.get_top_n_trigrams(insights, n=15)

            # Format IG phrases
            ig_phrases = []
            for word, score in zip(words, scores):
                ig_phrases.append((tuple(word.split(' ')), score))

            # Merge phrases
            merged_ig = ig_phrases
            merged_ngrams = top_bigrams + top_trigrams
            merged_all = ig_phrases + top_bigrams

            # Store results
            cluster_phrases['merged_phrases_all'][cluster_number] = merged_all
            cluster_phrases['merged_phrases_ig'][cluster_number] = merged_ig
            cluster_phrases['merged_phrases_ngrams'][cluster_number] = merged_ngrams
            cluster_phrases['top_words'][cluster_number] = top_words
            cluster_phrases['top_bigrams'][cluster_number] = top_bigrams
            cluster_phrases['top_trigrams'][cluster_number] = top_trigrams

        return cluster_phrases

    async def generate_cluster_label(self, insight_topics):
        """Generate intelligent cluster labels using OpenAI"""
        prompt = """Given the following list of topics extracted from a conversation, generate a concise and meaningful label (3–6 words) that best represents the main theme of the conversation. Avoid generic words. Focus on capturing the core subject or intent. The label should represent problems in these conversation whether about agent or user.


        Give a 2-4 words different labels that represents the theme of the conversation. The label should be specific and relevant to the topics discussed.

        Topics: {topics}

        Return a json object with the following structure:
        {{
            "label": "<generated_label1>
        }}
        """

        formatted_topics = "-".join(f"- {topic}" for topic in insight_topics)
        full_prompt = prompt.format(topics=formatted_topics)
        self.async_label_agent.prompt = full_prompt
        try:
            response = await self.async_label_agent.generate_response(json_resp=True)
            label = response["label"]
            return label
        except Exception as e:
            print(f"Error generating label: {e}")

    async def generate_cluster_label_with_kmeans_sampling(self, grouped_embeddings, grouped_summary, k=5):
        """Generate cluster labels using KMeans sampling for large clusters"""

        # Sample summaries using KMeans
        sampled_summaries = {}
        for cluster, embeddings in grouped_embeddings.items():
            summaries = grouped_summary[cluster]

            # If cluster has fewer items than k, use all summaries
            if len(embeddings) < k:
                sampled_summaries[cluster] = summaries
                continue

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(embeddings)
            centroids = kmeans.cluster_centers_

            # Find closest summaries to centroids
            sampled_indices = []
            for centroid in centroids:
                distances = np.linalg.norm(embeddings - centroid, axis=1)
                closest_idx = np.argmin(distances)
                sampled_indices.append(closest_idx)

            sampled_summaries[cluster] = [summaries[i]
                                          for i in sampled_indices]

        # Generate labels for each cluster
        cluster_labels = {}
        for cluster, summaries in tqdm(sampled_summaries.items()):
            prompt = """You are a conversation analysis expert helping label clusters of cold calling debt management calls.
Each cluster contains summaries of real customer conversations that were not converted due to some reason or issues.
Your task: Write a short, descriptive label (2 to 4 words) that best captures the common topic across the summaries in the cluster.
---
Summaries:
{summary}
Label:"""

            formatted_prompt = prompt.format(summary='\n'.join(summaries))
            self.async_label_agent.prompt = formatted_prompt

            try:
                response = await self.async_label_agent.generate_response()
                cluster_labels[cluster] = response
            except Exception as e:
                print(f"Error generating label for cluster {cluster}: {e}")
                cluster_labels[cluster] = f"Cluster {cluster}"

        return cluster_labels

    async def get_cluster_labels(self, cluster_phrases, col_name, phrase_type='merged_phrases_all'):
        """Generate labels for all clusters"""
        # cluster_labels = {}
        phrases_dict = cluster_phrases.get(phrase_type, {})
        keywords_dict = {
            col_name: [],
            f"heading": [],
        }

        for cluster, topics in tqdm(phrases_dict.items(), desc="Generating labels"):
            label = await self.generate_cluster_label(topics, )
            keywords_dict[col_name].append(cluster)
            keywords_dict[f"heading"].append(label)
        return pd.DataFrame(keywords_dict)

    async def merge_cluster_labels(self, cluster_labels):
        """Merge cluster labels into a single dictionary"""
        prompt = """You are assisting with thematic clustering of call center conversations related to debt counseling.

Here is a list of labeled conversation clusters, each assigned a cluster number and a descriptive label:

{cluster_labels}
**Your task:**

1. Identify clusters with overlapping or highly similar themes.

2. Assign each group a very descriptive label if they contain more than 1 theme separate by / 


Return only the dictionary in json format. Do not explain.

Example format:

{{
	"Wrong Contact": [ list of cluster numbers]
	...
}}
	"""
        cluster_info = ""
        for cluster, label in cluster_labels.items():
                cluster_info += f"Cluster {cluster}: {label}\n"

        prompt = prompt.format(cluster_labels=cluster_info)
        self.async_label_agent.prompt = prompt
        response = await self.async_label_agent.generate_response(json_resp=True)
        return response


    async def get_cluster_labels_with_kmeans(self, grouped_embeddings, grouped_summary, col_name, k=5):
        """Generate labels using KMeans sampling method"""
        cluster_labels = await self.generate_cluster_label_with_kmeans_sampling(
            grouped_embeddings, grouped_summary, k
        )
        # cluster_labels_merged = await self.merge_cluster_labels(cluster_labels)
        # for label, cluster_list in cluster_labels_merged.items():
        #     for cluster in cluster_list:
        #         cluster_labels[cluster] = label
        
        # Format results as DataFrame
        keywords_dict = {
            col_name: [],
            f"heading": [],
        }

        for cluster, label in cluster_labels.items():
            keywords_dict[col_name].append(cluster)
            keywords_dict[f"heading"].append(label)

        return pd.DataFrame(keywords_dict)

    async def analyze_clusters(self, df, col_name, phrase_type: Literal['merged_phrases_all', 'merged_phrases_ig', 'merged_phrases_ngrams'] = 'merged_phrases_all'):
        """Complete cluster analysis pipeline"""
        print("Preparing cluster data...")
        clustered_insights = self.prepare_cluster_data(
            df, col_name)

        print("Extracting cluster phrases...")
        cluster_phrases = self.extract_cluster_phrases(clustered_insights)
        print("Cluster phrases extracted.")
        return await self.get_cluster_labels(cluster_phrases)

    async def analyze_clusters_with_kmeans(self, df: pd.DataFrame, col_name, k=5):
        """Complete cluster analysis pipeline using KMeans sampling"""
        print("Generating cluster labels using KMeans sampling...")
        grouped_embeddings = df.groupby(
            col_name)['embedding'].apply(list).to_dict()
        grouped_summary = df.groupby(col_name)['summary'].apply(list).to_dict()
        return await self.get_cluster_labels_with_kmeans(grouped_embeddings, grouped_summary, col_name, k)

    async def label_group(self, df: pd.DataFrame, col_name):
        grouped_headings = df.groupby(
            col_name)['heading'].apply(list).to_dict()

        return await self.get_group_label(grouped_headings, col_name)

    async def get_group_label(self, grouped_headings, col_name):

        group_labels = {}

        for cluster, headings in tqdm(grouped_headings.items()):
            prompt = """You are helping analyze customer call transcripts related to debt counseling services. We have grouped multiple conversation clusters together based on thematic similarity.

Below are the labels of the individual clusters in this group:

{headings}

Your task is to generate a single high-level label that summarizes the shared theme of this group of clusters. The label should be:

- Clear and concise
- If more than one label applicable, separate them with a slash (/)
- Broad enough to encompass all the listed topics
- Specific enough to be meaningful in a call center analytics setting


**What is the best unifying label for this group? Return only the label.** 
                    """
            headings = " - ".join(headings)
            prompt = prompt.format(headings=headings)
            self.async_label_agent.prompt = prompt
            try:
                response = await self.async_label_agent.generate_response()
                group_labels[cluster] = response.strip()
                print(f"Group {cluster}: {response.strip()}")
            except Exception as e:
                print(f"Error generating label for group {cluster}: {e}")
                group_labels[cluster] = f"Group {cluster}"
        results_dict = {
            col_name:[],
            "group_heading": []
        }
        for cluster, label in group_labels.items():
            results_dict[col_name].append(cluster)
            results_dict["group_heading"].append(label)

        return pd.DataFrame(results_dict)


class TextClustering:
    """Main clustering algorithm implementation"""

    def __init__(self, config: TextClusteringConfig):
        self.config = config

        self.keyword_extractor = ClusterLabeler(config)
        self.embedding_model = config.embedding_model
        self.provider = config.provider
        print("Logging Model")
        self.embed = Embedding(
            embedding_model=self.embedding_model,
            provider=self.provider)

    def generate_embeddings(self, text_list, batch_size=1000):
        return self.embed.generate_embedding(text_list, batch_size)

    def _cluster_preprocess(self, df, prev_clusters=None):
        print(f"Data preprocessed. Shape: {df.shape}")

        # Limit data size
        if df.shape[0] > self.config.max_data_points:
            df = df.head(self.config.max_data_points)
            print(f"Limited data to {df.shape[0]} rows")

        # Clean and prepare data
        df.dropna(subset=["summary"], inplace=True)
        df.reset_index(inplace=True, drop=True)
        print(f"Total utterances to cluster: {df.shape[0]}")

        # Initialize clustering columns
        print("Clustering, ", df.shape[0])
        user_text_list = list(df["summary"])
        previous_text_list = list(
            prev_clusters["summary"]) if prev_clusters is not None else []
        previous_cluster_embeddings = self.generate_embeddings(
            previous_text_list) if prev_clusters is not None else []
        # map previous_text_list to embedding
        if prev_clusters is not None:
            prev_clusters["embedding"] = previous_cluster_embeddings

        user_text_embedding = self.generate_embeddings(user_text_list)
        df['embedding'] = user_text_embedding

        return df, user_text_embedding, prev_clusters, previous_cluster_embeddings

    def cluster_texts_analysewithth(self, df, user_text_embedding, prev_clusters=None, previous_cluster_embeddings=None, threshold=0.05):
        """Main clustering function"""
        print("Starting text clustering...")

        # Preprocess data

        # Process in chunks for memory efficiency
        df[f"confidence"] = 0
        df[f"cluster"] = 0
        total_data_length = df.shape[0]

        current_cluster = max(
            prev_clusters[f"cluster"])+1 if prev_clusters is not None else 1

        print(
            f"Total data length: {total_data_length}, Chunk size: {self.config.chunk_size}")
        for i in range(0, df.shape[0], self.config.chunk_size):
            print(f"Processing chunk {i//self.config.chunk_size + 1}")

            message_embeddings = user_text_embedding[i:i +
                                                     self.config.chunk_size]

            end_length = min(i + self.config.chunk_size, total_data_length)

            for index in range(i, end_length):
                if df.loc[index, f"cluster"] == 0:
                    if prev_clusters is not None:
                        prev_corr = np.inner(
                            message_embeddings[index - i], previous_cluster_embeddings)
                        for corr_column_index, corr_column_value in enumerate(prev_corr):
                            if corr_column_value >= threshold:
                                actual_index = corr_column_index
                                if corr_column_value > df.loc[index, f"confidence"]:
                                    df.loc[index, f"cluster"] = prev_clusters.loc[actual_index, f"cluster"]
                                    df.loc[index,
                                           f"confidence"] = corr_column_value

                    if df.loc[index, f"cluster"] == 0:
                        df.loc[index, f"cluster"] = current_cluster
                        df.loc[index, f"confidence"] = 1.0

                        # Calculate similarities within chunk
                        corr = np.inner(
                            message_embeddings[index - i], message_embeddings)

                        for corr_column_index, corr_column_value in enumerate(corr):
                            if corr_column_value >= threshold:
                                actual_index = corr_column_index + i

                                if df.loc[actual_index, f"cluster"] == 0:
                                    df.loc[actual_index,
                                           f"cluster"] = current_cluster
                                    df.loc[actual_index,
                                           f"confidence"] = corr_column_value
                                elif df.loc[actual_index, f"confidence"] <= corr_column_value:
                                    df.loc[actual_index,
                                           f"cluster"] = current_cluster
                                    df.loc[actual_index,
                                           f"confidence"] = corr_column_value

                        current_cluster += 1

        cluster_counts = df[f'cluster'].value_counts()
        valid_clusters = cluster_counts[cluster_counts >=
                                        self.config.min_cluster_size].index
        df = df[df[f'cluster'].isin(valid_clusters)]
        print(
            f"Clustering complete. Found {len(valid_clusters)} valid clusters")
        # Merge with previous clusters if available
        if prev_clusters is not None:
            df = pd.concat(
                [prev_clusters[['summary', 'embedding', f'cluster', f'confidence']], df], ignore_index=True)
            df.reset_index(drop=True, inplace=True)

            print(
                f"Total clusters after merging with previous: {df[f'cluster'].nunique()}")
        return df

    def add_cluster_size(self, df, col_name):
        """Add cluster size information"""
        cluster_sizes = df[col_name].value_counts().to_dict()
        df[f'cluster_size'] = df[col_name].map(cluster_sizes)
        return df

    def get_cluster_groups(self, cluster_df, similarity_df, col_name, prev_summary=None):
        """Group similar clusters together"""
        print("Grouping similar clusters...")
        # head 30 dict with key group no
        prev_group_dict = {}
        grcol = f'group'
        if prev_summary is not None:
            prev_group = prev_summary[[grcol, 'summary']]

            for index, row in prev_group.iterrows():
                group = row[grcol]
                if group not in prev_group_dict:
                    prev_group_dict[group] = []
                prev_group_dict[group].append(row['summary'])

        # Filter clusters by minimum size
        similarity_df = similarity_df[
            similarity_df[f"cluster_size"] >= self.config.min_cluster_size
        ]

        if similarity_df.shape[0] == 0:
            print("No clusters meet minimum size requirement")
            return similarity_df

        # Create embeddings for cluster representatives
        print("Encoding cluster data for grouping...")
        start_time = time.time()

        cluster_texts = []
        cluster_indices = {}

        for cluster in similarity_df[col_name]:
            cluster_texts_sample = cluster_df[
                cluster_df[col_name] == cluster
            ]["summary"].head(30).tolist()
            cluster_texts.extend(cluster_texts_sample)
            cluster_indices[cluster] = len(cluster_texts_sample)

        if not cluster_texts:
            return similarity_df

        cluster_embeddings = self.generate_embeddings(cluster_texts)

        print(f"Time taken to encode: {(time.time() - start_time)/60:.2f} min")

        # Initialize grouping
        similarity_df = similarity_df.copy()
        similarity_df[grcol] = 0
        similarity_df[f"group_similarity"] = 0.0

        current_group = 1

        # Create cluster embedding lookup
        embedding_start = 0
        cluster_embedding_map = {}

        for cluster in similarity_df[col_name]:
            size = cluster_indices[cluster]
            cluster_embedding_map[cluster] = cluster_embeddings[
                embedding_start:embedding_start + size
            ]
            embedding_start += size

        # Group clusters
        for idx, row in similarity_df.iterrows():
            cluster = row[col_name]

            if similarity_df.loc[idx, grcol] == 0:
                similarity_df.loc[idx, grcol] = current_group

                # Find similar clusters
                cluster_emb = cluster_embedding_map[cluster]

                for other_idx, other_row in similarity_df.iterrows():
                    if other_idx <= idx or similarity_df.loc[other_idx, grcol] != 0:
                        continue

                    other_cluster = other_row[col_name]
                    other_emb = cluster_embedding_map[other_cluster]

                    # Calculate similarity between clusters
                    similarities = np.inner(cluster_emb, other_emb)
                    max_sim = np.max(similarities)

                    if max_sim >= self.config.group_sim_cutoff:
                        if max_sim > similarity_df.loc[other_idx, f"group_similarity"]:
                            similarity_df.loc[other_idx,
                                              grcol] = current_group
                            similarity_df.loc[other_idx,
                                              f"group_similarity"] = max_sim

                current_group += 1

        return similarity_df

    async def run_clustering(self, df, prev_clustered=None, prev_grouped=None, prev_summary=None):
        """Complete clustering pipeline"""
        start_time = time.time()

        # Step 1: Cluster texts
        df_embedding = None
        prev_embedding = None
        df, df_embedding, prev_clustered, prev_embedding = self._cluster_preprocess(
            df, prev_clustered)
        col_name = f'cluster'

        clustered_df = self.cluster_texts_analysewithth(
            df, df_embedding, prev_clustered, prev_embedding, self.config.sim_threshold)

        print("Generating 2D Embedding")
        generated_2d_embeddings = self.generate_embeddings_2d_cached(
            clustered_df['embedding'].to_list())
        generated_2d_embeddings = [np.array(embedding) for embedding in generated_2d_embeddings]
        clustered_df['embedding2d'] = generated_2d_embeddings

        # Step 2: Extract keywords
        print("Extracting cluster keywords...")
        if self.config.label_with == 'topics':

            keywords_df = await self.keyword_extractor.analyze_clusters(
                clustered_df, phrase_type='merged_phrases_all', col_name=f'cluster')
        else:
            #cluster no and heading
            keywords_df = await self.keyword_extractor.analyze_clusters_with_kmeans(clustered_df, col_name='cluster', k=5)
            print(keywords_df.columns)
        # rename the cluster



        # Step 3: Add cluster sizes
        print("Adding cluster size information...")
        clustered_df = self.add_cluster_size(clustered_df, col_name=col_name)

        # Step 4: Create similarity dataframe
        similarity_df = clustered_df.groupby(col_name).agg({
            f'cluster_size': 'first',
            'summary': 'first'  # Take first text as representative
        }).reset_index()

        # Merge with keywords
        similarity_df = similarity_df.merge(
            keywords_df, on=col_name, how='left')
        similarity_df[f'heading'] = similarity_df[f'heading'].fillna('unknown')

        # Step 5: Group similar clusters
        grouped_df = self.get_cluster_groups(
            clustered_df, similarity_df, col_name=f'cluster', prev_summary=prev_summary)
        clustered_df2 = clustered_df.copy()
        # write group for each cluster in clustered_df
        for index, row in grouped_df.iterrows():
            cluster = row[f'cluster']
            group = row[f'group']
            clustered_df2.loc[clustered_df2[f'cluster']
                              == cluster, f'group'] = group

        # Add group keywords to grouped_df
        keywords_group = await self.keyword_extractor.label_group(
            grouped_df, col_name=f'group')

        # merge keywords with grouped_df

        grouped_df = grouped_df.merge(keywords_group, on=f'group', how='left')
        print(grouped_df.columns)
        print(clustered_df.columns)
        # self.save_results(clustered_df2, grouped_df)
        summary_report = self.create_summary_report(clustered_df, grouped_df)
        summary_report.to_parquet(
            f"{self.config.output_path}/summary_report.parquet", index=False, engine='pyarrow')

        total_time = (time.time() - start_time) / 60
        print(f"Clustering completed in {total_time:.2f} minutes")

    def save_results(self, clustered_df, grouped_df):
        """Save clustering results"""
        try:
            # Save individual clusters
            clustered_df.to_excel(
                self.config.output_path / 'clustered_texts1.xlsx',
                index=False
            )

            # Save cluster groups
            grouped_df.to_excel(
                self.config.output_path / 'cluster_groups1.xlsx',
                index=False
            )

            # Create summary report
            summary_report = self.create_summary_report(
                clustered_df, grouped_df)

            print(f"Results saved to {self.config.output_path}")

        except Exception as e:
            print(f"Error saving results: {e}")

    def create_summary_report(self, clustered_df: pd.DataFrame, grouped_df):
        """Create a summary report of clustering results"""
        # Merge cluster data with group information
        summary = clustered_df.merge(grouped_df[[f'cluster', f'group', 'group_heading', f'heading']],
                                     on=f'cluster', how='left')

        # summary = summary[[f'group', f'cluster', f'heading', 'summary', f'transcript',
        #                    f'confidence', f'cluster_size', 'embedding', 'embedding2d']]
        # headings = summary['heading'].unique().tolist()
        # new_cluster_map = {heading: i+1 for i, heading in enumerate(headings)}

        # for index, row in summary.iterrows():
        #     heading = row['heading']
        #     if heading in new_cluster_map:
        #         summary.at[index, 'cluster'] = new_cluster_map[heading]

        return summary

    def generate_embeddings_2d_cached(self, embeddings):
        """Generate 2D embeddings."""
        embeddings_2d = reduce_embeddings(
            embeddings, method=self.config.embedding_reduce_method, n_components=2)
        return embeddings_2d

async def main():
    """Example usage of the clustering algorithm"""
    output_path = '/home/ori/Desktop/conveval/conval/c'
    config = TextClusteringConfig(
        sim_threshold=0.70,
        group_sim_cutoff=0.8,
        min_cluster_size=3,
        output_path= output_path,
        embedding_model= 'Qwen/Qwen3-Embedding-8B',
        provider='vllm',
        label_agent= "gpt-4.1",
        label_provider= "openai",
    )
    df = pd.read_json('/home/ori/Desktop/conveval/nc.json')
    clustering = TextClustering(config)
    await clustering.run_clustering(df, prev_clustered=None,
                              prev_grouped=None, prev_summary=None)
if __name__ == "__main__":
   asyncio.run(main())
