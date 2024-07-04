import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class TemporalLobe:
    def __init__(self):
        """
        Initializes the TemporalLobe class with predefined auditory keywords and machine learning models.
        """
        self.embedding_dim = 50  # Dimensionality of word embeddings
        self.auditory_keywords = ['hear', 'listen', 'sound', 'music', 'noise', 'silent', 'volume', 'whisper', 'shout', 'speak']
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer, self.model)
        self.word_embeddings = self._initialize_word_embeddings()

        initial_data = ["I hear music", "Listen to the sound", "The sound is loud"]
        initial_labels = [0, 1, 0]
        self.pipeline.fit(initial_data, initial_labels)

        self._load_model()

    def _initialize_word_embeddings(self):
        """
        Initializes word embeddings for the auditory keywords and other words.

        Returns:
            dict: A dictionary of word embeddings for the auditory keywords.
        """
        vocabulary = self.auditory_keywords + ['the', 'is', 'a', 'to', 'and', 'can', 'some', 'this', 'room', 'level', 'high']
        embeddings = {word: np.random.rand(self.embedding_dim) for word in vocabulary}
        return embeddings

    def _load_model(self):
        """
        Loads the machine learning model from a file if available.
        """
        try:
            with open('temporal_lobe_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
        except FileNotFoundError:
            pass

    def _save_model(self):
        """
        Saves the machine learning model to a file.
        """
        with open('temporal_lobe_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)

    def process(self, prompt: str) -> str:
        """
        Processes the given prompt to analyze auditory content.

        Args:
            prompt (str): The input sentence to be processed.

        Returns:
            str: A detailed response summarizing the analysis of the prompt.
        """
        try:
            features = self._extract_features(prompt)
            prediction = self.pipeline.named_steps['multinomialnb'].predict(features)
            auditory_analysis = self._analyze_auditory_content(prompt)
            self._train_model(prompt, auditory_analysis)
            return f"Auditory analysis complete. {auditory_analysis} Full analysis: {auditory_analysis}"
        except Exception as e:
            return f"Error processing temporal lobe: {e}"

    def _extract_features(self, prompt: str) -> Any:
        """
        Extracts feature vectors from the given prompt using CountVectorizer.

        Args:
            prompt (str): The input sentence to be vectorized.

        Returns:
            scipy.sparse.csr.csr_matrix: The feature vectors extracted from the prompt.
        """
        return self.pipeline.named_steps['countvectorizer'].transform([prompt])

    def _analyze_auditory_content(self, prompt: str) -> str:
        """
        Analyzes the prompt for auditory content based on predefined auditory keywords.

        Args:
            prompt (str): The input sentence to be analyzed.

        Returns:
            str: A summary of auditory elements detected in the prompt.
        """
        words = prompt.lower().split()
        auditory_words = [word for word in words if word in self.auditory_keywords]
        
        # Check for similar words if no direct keywords are found
        if not auditory_words:
            for word in words:
                if self._find_similar_word(word):
                    auditory_words.append(word)
        
        if auditory_words:
            return f"Auditory elements detected: {', '.join(auditory_words)}"
        return "No explicit auditory elements detected"

    def _find_similar_word(self, word: str) -> bool:
        """
        Finds a similar word in the auditory keywords using cosine similarity.

        Args:
            word (str): The word to find a similar word for.

        Returns:
            bool: True if a similar word is found, False otherwise.
        """
        if word not in self.word_embeddings:
            return False
        
        word_embedding = self.word_embeddings[word]
        similarities = {kw: self._cosine_similarity(word_embedding, self.word_embeddings[kw]) for kw in self.auditory_keywords}
        most_similar_word = max(similarities, key=similarities.get)
        
        return similarities[most_similar_word] > 0.7  # Threshold for considering words as similar

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes the cosine similarity between two vectors.

        Args:
            vec1 (np.ndarray): The first vector.
            vec2 (np.ndarray): The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _train_model(self, prompt: str, analysis: str):
        """
        Trains the model incrementally with the new prompt and its analysis.

        Args:
            prompt (str): The input sentence.
            analysis (str): The analysis result of the prompt.
        """
        labels = [1 if "detected" in analysis else 0]
        feature_vector = self._extract_features(prompt)
        self.pipeline.named_steps['multinomialnb'].partial_fit(feature_vector, labels, classes=np.array([0, 1]))
        self._update_word_embeddings(prompt)
        self._save_model()

    def _update_word_embeddings(self, prompt: str):
        """
        Updates word embeddings based on the context of the given prompt.

        Args:
            prompt (str): The input sentence.
        """
        words = prompt.lower().split()
        context_window = 2  # Number of words to consider as context on each side

        for i, word in enumerate(words):
            if word in self.word_embeddings:
                context_words = words[max(0, i - context_window): i] + words[i + 1: i + 1 + context_window]
                for context_word in context_words:
                    if context_word in self.word_embeddings:
                        self.word_embeddings[word] += 0.01 * (self.word_embeddings[context_word] - self.word_embeddings[word])

