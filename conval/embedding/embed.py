import openai
from google import genai
from google.genai import types
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
from typing import Literal, Union, Optional, List
import pandas as pd
import numpy as np
import json
from umap import umap_
from openai import OpenAI

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
class Embedding:
    """
    A class to generate embeddings for text using various providers like Google, OpenAI, and Sentence Transformers.
    This class supports generating embeddings for a list of texts.
    It can handle different providers and models based on the user's choice.
    Attributes:
        embedding_model (str): The model to use for generating embeddings.
        provider (Provider): The provider to use for generating embeddings.
    Methods:
        generate_embedding(text: str) -> np.ndarray:
            Generates an embedding for the given text using the specified model and provider.
        _google_embedding(text: str, batch_size: int = 100) -> list[float]:
            Generates embeddings using Google API.
        _openai_embedding(text_list: str, batch_size: int = 100) -> list[float]:
            Generates embeddings using OpenAI API.
        _sentence_transformers_embedding(text_list: str, batch_size: int = 100) -> list[float]:
            Generates embeddings using Sentence Transformers.
    Usage:
        embedding = Embedding(embedding_model="text-embedding-ada-002", provider="openai")
        embeddings = embedding.generate_embedding(["Hello world", "This is a test"])
    
        
    """
    def __init__(self, embedding_model: str, provider: Literal['google', 'openai', 'sentence_transformers']):
        self.provider = provider
        # 3 provider ['google','openai', sentence_transformers']
        if self.provider == 'google':
            self.model = embedding_model
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        elif self.provider == 'openai':
            self.model = embedding_model
            self.client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == 'sentence_transformers':
            self.model = embedding_model
            from sentence_transformers import SentenceTransformer
            self.client = SentenceTransformer(self.model)
        elif self.provider == 'vllm':
            self.model = embedding_model
            self.client = OpenAI(base_url=os.getenv("VLLM_ENDPOINT"), 
            api_key=os.getenv("VLLM_API_KEY"))

            print(f"Using VLLM with model {self.model} at {os.getenv('VLLM_ENDPOINT')}")
    
    def generate_embedding(self, text: str, batch_size: int=100) -> List[np.ndarray]:
        """
        Generate an embedding for the given text using the specified model and provider.
        
        :param text: The input text to generate an embedding for.
        :param batch_size: The size of the batch for processing the text.
        :return: A numpy array representing the embedding.
        """
        if self.provider == 'google':
            # Call Google embedding API
            return self._google_embedding(text, batch_size)
        elif self.provider == 'openai':
            # Call OpenAI embedding API
            return self._openai_embedding(text, batch_size)
        elif self.provider == 'sentence_transformers':
            return self._sentence_transformers_embedding(text, batch_size)
        elif self.provider == 'vllm':
            return self._openai_embedding(text, batch_size)
    
    def _google_embedding(self, text: str, batch_size:int =100) -> list[float]:
        # Placeholder for Google embedding logic
        all_embeddings = []
        print(len(text))
        if not text:
            return all_embeddings
        
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i:i + batch_size]
            embedding = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="CLUSTERING")
            )

            all_embeddings.extend(embedding.embeddings)
        all_embeddings = [np.array(emb.values) for emb in all_embeddings]
        return all_embeddings

    def _openai_embedding(self, text_list: str, batch_size:int=100) -> list[float]:
        all_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def _sentence_transformers_embedding(self, text_list: str, batch_size:int=100) -> list[float]:
        all_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i:i + batch_size]
            batch_embeddings = self.client.encode(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def __repr__(self):
        return f"Embedding({self.embedding})"
    
def reduce_embeddings(embeddings: Union[list[list[float]], np.ndarray],
    method: Literal["pca", "tsne", "umap"] = "pca",
    n_components: int = 2,
    **kwargs) -> np.ndarray:
        """
        Reduce the dimensionality of embeddings using PCA, t-SNE, or UMAP.

        Args:
            embeddings: List or np.ndarray of shape (n_samples, n_features).
            method: Reduction method: "pca", "tsne", or "umap".
            n_components: Number of dimensions to reduce to (usually 2 or 3).
            **kwargs: Additional keyword arguments for the selected method.

        Returns:
            np.ndarray of shape (n_samples, n_components).
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if method == "pca":
            reducer = PCA(n_components=n_components, **kwargs)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, **kwargs)
        elif method == "umap":
            reducer = umap_.UMAP(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        reduced = reducer.fit_transform(embeddings)
        return reduced

def plot_embeddings(
    reduced_embeddings: Union[np.ndarray, list[list[float]]],
    labels: Optional[list[str]] = None,
    title: str = "Embedding Visualization",
    figsize: tuple = (8, 6),
    cmap: str = "tab10",
    alpha: float = 0.7,
    save_path: Optional[str] = None
):
    """
    Plot 2D embeddings with optional labels (as colors).

    Args:
        reduced_embeddings: 2D embeddings of shape (n_samples, 2).
        labels: Optional list of labels for coloring the points.
        title: Title of the plot.
        figsize: Size of the figure.
        cmap: Matplotlib colormap to use.
        alpha: Point transparency.
        save_path: If provided, save the plot to this path.
    """
    if not isinstance(reduced_embeddings, np.ndarray):
        reduced_embeddings = np.array(reduced_embeddings)

    assert reduced_embeddings.shape[1] == 2, "Embeddings must be 2D for plotting"

    plt.figure(figsize=figsize)

    if labels is not None:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            idx = labels == label
            plt.scatter(
                reduced_embeddings[idx, 0],
                reduced_embeddings[idx, 1],
                label=str(label),
                alpha=alpha,
                color=colors[i],
                edgecolors='k',
                linewidths=0.3
            )
        plt.legend()
    else:
        plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            alpha=alpha,
            edgecolors='k',
            linewidths=0.3
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
