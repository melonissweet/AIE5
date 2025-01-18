import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import os
import faiss
from datetime import datetime
# from openai_utils.embedding import EmbeddingModel
# from dotenv import load_dotenv
# load_dotenv() 


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array) -> None:
        self.vectors[key] = vector

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[str, float]]:
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self

"""
Activity:
- Implement a new distance metric
- Add metadata support to the vector database
Created new class VectorDatabase_Ann to test ANN as distance metrics and to insert metadata to vector database
"""

def ann_similarity(query_vector: np.array, vectors: Dict[str, Dict], k: int=3) -> List[Tuple[str, float]]:
    """Using FAISS library to use ANN to find the k most similar vectors to the query_vector"""
    dimension = query_vector.shape[0]
    index = faiss.IndexFlatIP(dimension)  # Using inner product (dot product) for similarity
    all_vectors = np.array([item['vector'] for item in vectors.values()])
    index.add(all_vectors.astype('float32'))
    query_vector = np.expand_dims(query_vector, axis=0).astype('float32')
    distances, indices = index.search(query_vector, k)
    keys = list(vectors.keys())
    return [(keys[i], distances[0][j]) for j, i in enumerate(indices[0])]


class VectorDatabase_Ann:
    def __init__(self, embedding_model: EmbeddingModel = None,):
        self.vectors = defaultdict(lambda: {'vector': np.array([]), 'metadata': {}})
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, meta: Dict) -> None:
        self.vectors[key]['vector'] = vector 
        self.vectors[key]['metadata'] = meta

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = ann_similarity,
    ) -> List[Tuple[str, float]]:
        return distance_measure(query_vector, self.vectors, k)

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = ann_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        if isinstance(query_vector, list): # Ensure query_vector is a numpy array 
            query_vector = np.array(query_vector)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str], file_path: str, run_datetime) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        file_name = os.path.basename(file_path).replace('.pdf', '')
        metadata = {"title": file_name,
                    "run_datetime": run_datetime}
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding), metadata)
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]
    file_path = 'data/test.pdf'

    run_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    vector_db = VectorDatabase_Ann()
    print('Created the VectorDatabase_Ann instance')
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text, file_path, run_datetime))
    print('Created embedding and inserted to VectorDatabase')
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
