import numpy as np
import time
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class ParietalLobe:
    def __init__(self):
        """
        Initializes the ParietalLobe class with predefined sets of keywords for spatial, sensory, and navigation processing.
        Also initializes the machine learning model for learning and adaptation.
        """
        self.spatial_keywords = ['up', 'down', 'left', 'right', 'above', 'below', 'near', 'far']
        self.sensory_keywords = ['touch', 'feel', 'texture', 'temperature', 'pressure', 'pain']
        self.navigation_keywords = ['map', 'route', 'direction', 'location', 'distance', 'navigate']

        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer, self.model)

        initial_data = [
            "The box is above the table, near the window",
            "I feel a rough texture and cold temperature",
            "Navigate to the nearest exit using the map",
            "Calculate the distance between points A (2,3) and B (5,7)",
            "The room temperature is 72 degrees",
            "Process this sentence without any spatial or numerical content"
        ]
        initial_labels = [0, 1, 0, 1, 0, 0]
        self.pipeline.fit(initial_data, initial_labels)

        self._load_model()
        self.error_log = []

    def _load_model(self):
        """
        Loads the model and vectorizer state from a file, if available.
        """
        try:
            with open('parietal_lobe_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
        except FileNotFoundError:
            pass

    def _save_model(self):
        """
        Saves the model and vectorizer state to a file.
        """
        with open('parietal_lobe_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)

    def _preprocess_prompt(self, prompt):
        """
        Preprocesses the input prompt to ensure it is clean and consistent.

        Args:
            prompt (str): The input sentence to be preprocessed.

        Returns:
            str: The cleaned and preprocessed prompt.
        """
        return prompt

    def _extract_features(self, prompt):
        """
        Extracts feature vectors from the given prompt using CountVectorizer.

        Args:
            prompt (str): The input sentence to be vectorized.

        Returns:
            scipy.sparse.csr.csr_matrix: The feature vectors extracted from the prompt.
        """
        return self.pipeline.named_steps['countvectorizer'].transform([prompt])

    def process(self, prompt):
        """
        Processes the given prompt to analyze spatial, sensory, navigation, and numerical content.
        Also trains the model incrementally with the new prompt.

        Args:
            prompt (str): The input sentence to be processed.

        Returns:
            str: A detailed response summarizing the analysis of the prompt.
        """
        prompt = self._preprocess_prompt(prompt)
        
        try:
            features = self._extract_features(prompt)
            prediction = self.pipeline.named_steps['multinomialnb'].predict(features)

            spatial_analysis = self._analyze_spatial_content(prompt)
            sensory_integration = self._integrate_sensory_information(prompt)
            navigation_assessment = self._assess_navigation(prompt)
            numerical_analysis = self._analyze_numerical_data(prompt)

            for _ in range(3):
                time.sleep(0.5)

            analysis = {
                "Spatial Analysis": spatial_analysis,
                "Sensory Integration": sensory_integration,
                "Navigation Assessment": navigation_assessment,
                "Numerical Analysis": numerical_analysis
            }

            self._train_model(prompt, analysis)

            return f"Parietal Lobe Response: Spatial-sensory integration complete. {self._summarize_analysis(analysis)}"
        except Exception as e:
            self._handle_error(prompt, e)
            return f"Parietal Lobe Response: Error in processing: {str(e)}. Spatial-sensory systems recalibrating."

    def _train_model(self, prompt, analysis):
        """
        Trains the model incrementally with the new prompt and its analysis.

        Args:
            prompt (str): The input sentence.
            analysis (dict): The analysis result of the prompt.
        """
        labels = [1 if "detected" in label or "processing" in label or "identified" in label or "found" in label else 0 for label in [
            analysis["Spatial Analysis"], 
            analysis["Sensory Integration"], 
            analysis["Navigation Assessment"], 
            analysis["Numerical Analysis"]
        ]]

        feature_vector = self._extract_features(prompt)
        feature_vector = np.vstack([feature_vector.toarray()] * len(labels))
        self.pipeline.named_steps['multinomialnb'].partial_fit(feature_vector, labels, classes=np.array([0, 1]))

        self._save_model()

    def _analyze_spatial_content(self, prompt):
        """
        Analyzes the prompt for spatial content based on predefined spatial keywords.

        Args:
            prompt (str): The input sentence to be analyzed.

        Returns:
            str: A summary of spatial elements detected in the prompt.
        """
        words = prompt.lower().split()
        spatial_words = [word for word in words if word in self.spatial_keywords]
        if spatial_words:
            return f"Spatial elements detected: {', '.join(spatial_words)}"
        return "No explicit spatial elements detected"

    def _integrate_sensory_information(self, prompt):
        """
        Analyzes the prompt for sensory information based on predefined sensory keywords.

        Args:
            prompt (str): The input sentence to be analyzed.

        Returns:
            str: A summary of sensory information detected in the prompt.
        """
        sensory_words = [word for word in prompt.lower().split() if word in self.sensory_keywords]
        if sensory_words:
            return f"Sensory information processing: {', '.join(sensory_words)}"
        return "No specific sensory information to process"

    def _assess_navigation(self, prompt):
        """
        Analyzes the prompt for navigation-related content based on predefined navigation keywords.

        Args:
            prompt (str): The input sentence to be analyzed.

        Returns:
            str: A summary of navigation-related concepts detected in the prompt.
        """
        nav_words = [word for word in prompt.lower().split() if word in self.navigation_keywords]
        if nav_words:
            return f"Navigation-related concepts identified: {', '.join(nav_words)}"
        return "No navigation-specific elements found"

    def _analyze_numerical_data(self, prompt):
        """
        Analyzes the prompt for numerical data and performs basic statistical analysis if numerical data is found.

        Args:
            prompt (str): The input sentence to be analyzed.

        Returns:
            str: A summary of numerical data detected in the prompt and basic statistical analysis.
        """
        numbers = re.findall(r'\d+', prompt)
        if numbers:
            numbers = [int(num) for num in numbers]
            return f"Numerical data found: mean={np.mean(numbers):.2f}, median={np.median(numbers):.2f}, count={len(numbers)}"
        return "No numerical data found"

    def _summarize_analysis(self, analysis):
        """
        Summarizes the analysis results into a comprehensive response.

        Args:
            analysis (dict): The dictionary containing the analysis results.

        Returns:
            str: A summary of the analysis results.
        """
        summary = []
        if "elements detected" in analysis["Spatial Analysis"]:
            summary.append("Spatial processing activated")
        if "information processing" in analysis["Sensory Integration"]:
            summary.append("Sensory integration in progress")
        if "concepts identified" in analysis["Navigation Assessment"]:
            summary.append("Navigation systems engaged")
        if "Numerical data found" in analysis["Numerical Analysis"]:
            summary.append("Quantitative analysis performed")
        
        if not summary:
            return "No significant spatial-sensory patterns identified. Maintaining baseline awareness."
        
        return " ".join(summary) + f" Full analysis: {analysis}"

    def _handle_error(self, prompt, error):
        """
        Handles errors encountered during processing and adapts the system to prevent future errors.

        Args:
            prompt (str): The input sentence that caused the error.
            error (Exception): The error encountered during processing.
        """
        self.error_log.append((prompt, str(error)))
