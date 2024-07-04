import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class WebResearchTool:
    def __init__(self, max_results=15, max_content_length=4000, progress_callback = None):
        self.max_results = max_results
        self.max_content_length = max_content_length
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.driver = self._setup_selenium()

    def _setup_selenium(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)
        return driver

    def web_research(self, query: str, progress_callback=None) -> str:
        try:
            search_results = self._search(query)
            with ThreadPoolExecutor(max_workers=self.max_results) as executor:
                future_to_url = {executor.submit(self._extract_content, url): url for url in search_results}
                contents = []
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content:
                            contents.append((url, content))
                    except Exception as exc:
                        print(f"{url} generated an exception: {exc}")

            if not contents:
                return f"No relevant information found for the query: {query}"

            summarized_content = self._summarize_content(query, contents)
            return summarized_content
        except Exception as e:
            return f"An error occurred during web research: {str(e)}. Please try a different query or check your internet connection."

    def _search(self, query: str) -> List[str]:
        url = f"https://www.google.com/search?q={query}&num={self.max_results}"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = []
        for g in soup.find_all('div', class_='g'):
            anchors = g.find_all('a')
            if anchors:
                link = anchors[0]['href']
                if link.startswith('http'):
                    search_results.append(link)
                    if len(search_results) == self.max_results:
                        break
        return search_results

    def _extract_content(self, url: str) -> str:
        try:
            self.driver.get(url)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
            return content
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return ""

    def _summarize_content(self, query: str, contents: List[Tuple[str, str]]) -> str:
        all_sentences = []
        for url, content in contents:
            sentences = sent_tokenize(content)
            all_sentences.extend([(sentence, url) for sentence in sentences])

        if not all_sentences:
            return f"No relevant information found for the query: {query}"

        sentence_vectors = self.vectorizer.fit_transform([sentence for sentence, _ in all_sentences])
        
        num_clusters = min(5, len(all_sentences))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(sentence_vectors)
        
        cluster_sentences = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(kmeans.labels_):
            cluster_sentences[label].append(all_sentences[idx])
        
        summary = self._format_summary(query, cluster_sentences)
        return summary

    def _format_summary(self, query: str, cluster_sentences: List[List[Tuple[str, str]]]) -> str:
        formatted_summary = f"Web research results for '{query}':\n\n"
        
        for i, cluster in enumerate(cluster_sentences, 1):
            if cluster:
                representative_sentence = max(cluster, key=lambda x: len(x[0]))
                sentiment = self.sentiment_analyzer.polarity_scores(representative_sentence[0])
                sentiment_summary = f" (Sentiment: {sentiment})"
                formatted_summary += f"{i}. {representative_sentence[0]}{sentiment_summary}\n"
                for sentence, url in cluster[:2]:  # Include up to 2 supporting sentences
                    if sentence != representative_sentence[0]:
                        formatted_summary += f"   - {sentence}\n"
                formatted_summary += "\n"
        
        # Add sources
        formatted_summary += "Sources:\n"
        unique_urls = set()
        for cluster in cluster_sentences:
            for _, url in cluster:
                if url not in unique_urls:
                    unique_urls.add(url)
                    formatted_summary += f"- {url}\n"
                if len(unique_urls) >= 10:  # Limit to 10 sources
                    break
            if len(unique_urls) >= 10:
                break
        
        # Ensure the summary is within max_content_length
        if len(formatted_summary) > self.max_content_length:
            formatted_summary = formatted_summary[:self.max_content_length] + '...'
        
        return formatted_summary

    def __del__(self):
        self.driver.quit()

# if __name__ == "__main__":
#     research_tool = WebResearchTool()
#     result = research_tool.web_research("Latest advancements in artificial intelligence")
#     print(result)
