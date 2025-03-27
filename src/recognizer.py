from src.utils.feature_extraction import extract_features
from src.utils.image_loader import get_all_image_paths
from src.utils.similarity import cosine_similarity
import os
import numpy as np


class Recognizer:
    def __init__(self, dataset_path, use_cache=True):
        self.dataset_path = dataset_path
        self.embeddings = {}
        self.cache_path = "data/embeddings/banco_embeddings.npy"
        self.use_cache = use_cache
        self._load_embeddings()

    def _load_embeddings(self):
        if self.use_cache and os.path.exists(self.cache_path):
            print("Carregando embeddings do cache...")
            self.embeddings = np.load(
                self.cache_path, allow_pickle=True).item()
            return

        print("Extraindo embeddings das imagens...")
        for img_path in get_all_image_paths(self.dataset_path):
            embedding = extract_features(img_path)
            if embedding is not None:
                name = os.path.splitext(os.path.basename(img_path))[0]
                self.embeddings[name] = embedding

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.save(self.cache_path, self.embeddings)
        print(f"{len(self.embeddings)} embeddings salvos em {self.cache_path}")

    def recognize(self, img_path):
        target_embedding = extract_features(img_path)
        if target_embedding is None:
            return None, 0.0

        best_match = None
        best_score = -1

        for name, emb in self.embeddings.items():
            score = cosine_similarity(target_embedding, emb)
            if score > best_score:
                best_match = name
                best_score = score

        return best_match, best_score
