import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class WebResearchTool:
    def __init__(self, max_results=5, max_depth=2):
        self.max_results = max_results
        self.max_depth = max_depth
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()

    def search(self, query, progress_callback):
        progress_callback(f"Searching for: {query}")
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.select('.yuRUbf a')
        results = [result['href'] for result in search_results[:self.max_results]]
        progress_callback(f"Found {len(results)} search results")
        return results

    def extract_text(self, url, progress_callback):
        progress_callback(f"Extracting text from: {url}")
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style", "meta", "noscript"]):
                script.decompose()
            text = soup.get_text(separator=' ', strip=True)
            progress_callback(f"Successfully extracted text from: {url}")
            return text
        except Exception as e:
            progress_callback(f"Failed to extract text from: {url}. Error: {str(e)}")
            return ""

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return " ".join(tokens)

    def calculate_similarity(self, query, text):
        preprocessed_query = self.preprocess_text(query)
        preprocessed_text = self.preprocess_text(text)
        tfidf_matrix = self.vectorizer.fit_transform([preprocessed_query, preprocessed_text])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def extract_relevant_info(self, text, query, progress_callback):
        progress_callback("Extracting relevant information")
        sentences = nltk.sent_tokenize(text)
        relevant_sentences = []
        for sentence in sentences:
            similarity = self.calculate_similarity(query, sentence)
            if similarity > 0.2:  # Increased threshold for better relevance
                relevant_sentences.append(sentence)
        progress_callback(f"Extracted {len(relevant_sentences)} relevant sentences")
        return " ".join(relevant_sentences)

    def web_research(self, query, progress_callback):
        progress_callback("Starting web research")
        
        urls = self.search(query, progress_callback)
        results = []

        for url in urls:
            text = self.extract_text(url, progress_callback)
            if text:
                relevant_info = self.extract_relevant_info(text, query, progress_callback)
                if relevant_info:
                    results.append({
                        "url": url,
                        "content": relevant_info
                    })
                    progress_callback(f"Added relevant information from: {url}")

        if not results:
            progress_callback("No relevant information found")
            return f"Unable to find relevant information for the query: {query}"

        progress_callback("Aggregating results")

        aggregated_content = ""
        for result in results:
            aggregated_content += f"[Source: {result['url']}]\n{result['content']}\n\n"

        progress_callback("Web research completed")

        return aggregated_content.strip()