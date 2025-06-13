!pip install  keybert

import spacy
import nltk
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import string
from transformers import pipeline
import re
from IPython.display import display, HTML

from typing import List, Optional

class KeywordExtractor:
    """
    Keyword extraction system that dynamically adjusts output based on text length.
    Extracts approximately 5% of total words as keywords.
    """
    
    def __init__(self, 
                 spacy_model: str = "en_core_web_sm",
                 keybert_model: str = "paraphrase-mpnet-base-v2"):
        # Initialize models once for efficiency
        self.nlp = spacy.load(spacy_model)
        self.kw_model = KeyBERT(model=keybert_model)
    
    def _count_words(self, text: str) -> int:
        """Count meaningful words in text (excluding stop words and punctuation)."""
        doc = self.nlp(text)
        return len([token for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])
    
    def _calculate_target_keywords(self, text: str, percentage: float = 0.05) -> int:
        """Calculate target number of keywords based on text length."""
        word_count = self._count_words(text)
        target_keywords = max(5, int(word_count * percentage))  # Minimum 5 keywords
        return min(target_keywords, 100)  # Maximum 100 keywords for very long texts
    
    def extract_named_entities(self, text: str) -> List[str]:
        """Extract named entities."""
        doc = self.nlp(text)
        entity_types = ["ORG", "PRODUCT", "WORK_OF_ART", "PERSON", "GPE", 
                       "LOC", "FAC", "EVENT", "NORP", "LAW", "LANGUAGE"]
        entities = [ent.text.strip() for ent in doc.ents 
                   if ent.label_ in entity_types and len(ent.text.strip()) > 1]
        return list(set(entities))
    
    def extract_noun_chunks(self, text: str) -> List[str]:
        """Extract meaningful noun chunks."""
        doc = self.nlp(text)
        chunks = []
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if (len(chunk_text) >= 3 and 
                len(chunk_text.split()) > 1 and
                not chunk.root.pos_ in ['PRON']):
                chunks.append(chunk_text)
        return list(set(chunks))
    
    def extract_tfidf_keywords(self, text: str, top_n: int) -> List[str]:
        """Extract TF-IDF keywords."""
        try:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 10]
            
            if len(sentences) < 2:
                sentences = [text]
            
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=500,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            word_scores = list(zip(feature_names, scores))
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [word for word, score in word_scores[:top_n]]
        except:
            return []
    
    def extract_keybert_keywords(self, text: str, top_n: int) -> List[str]:
        """Extract KeyBERT keywords."""
        try:
            keywords = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_k=top_n,
                use_mmr=True,
                diversity=0.5
            )
            return [kw[0] for kw in keywords]
        except:
            return []
    
    def clean_and_deduplicate_keywords(self, keywords: List[str]) -> List[str]:
        """Remove duplicates and similar keywords."""
        if not keywords:
            return []
        
        cleaned = []
        seen = set()
        
        for kw in keywords:
            kw_clean = re.sub(r'[^\w\s]', '', kw.lower().strip())
            
            if len(kw_clean) < 2:
                continue
            
            # Check for exact matches and substring overlaps
            skip = False
            for seen_kw in seen:
                if (kw_clean == seen_kw or 
                    kw_clean in seen_kw or 
                    seen_kw in kw_clean):
                    skip = True
                    break
            
            if not skip:
                seen.add(kw_clean)
                cleaned.append(kw.strip())
        
        return cleaned



def extract_keywords(self, text: str, percentage: float = 0.15) -> List[str]:
        """
        Extract keywords based on text length.
        
        Args:
            text: Input text
            percentage: Percentage of total words to extract as keywords (default: 0.15 = 15%)
            
        Returns:
            List of keywords
        """
        if not text or len(text.strip()) < 50:
            return []
        
        # Calculate target number of keywords based on text length
        target_keywords = self._calculate_target_keywords(text, percentage)
        
        # Extract keywords using all methods
        # Adjust individual method limits based on target
        method_ratio = max(0.6, target_keywords / 25)  # Scale method outputs
        
        ner_keywords = self.extract_named_entities(text)
        noun_keywords = self.extract_noun_chunks(text)
        tfidf_keywords = self.extract_tfidf_keywords(text, int(15 * method_ratio))
        keybert_keywords = self.extract_keybert_keywords(text, int(20 * method_ratio))
        
        # Combine with weighted frequency
        weighted_freq = Counter()
        
        # Weight different methods
        for kw in ner_keywords:
            weighted_freq[kw] += 1.5  # Named entities are important
        for kw in noun_keywords:
            weighted_freq[kw] += 1.0
        for kw in tfidf_keywords:
            weighted_freq[kw] += 1.2
        for kw in keybert_keywords:
            weighted_freq[kw] += 1.3  # Semantic keywords are important
        
        # Get top keywords by weighted frequency
        top_keywords = [kw for kw, freq in weighted_freq.most_common(target_keywords * 2)]
        
        # Clean and deduplicate
        final_keywords = self.clean_and_deduplicate_keywords(top_keywords)
        
        # Return exactly the target number of keywords
        return final_keywords[:target_keywords]

# Simple usage function
def extract_keywords_from_text(text: str, percentage: float = 0.15) -> List[str]:
    """
    Simple function to extract keywords from text.
    
    Args:
        text: Input text/article
        percentage: Percentage of words to extract as keywords (0.15 = 15%)
        
    Returns:
        List of keywords
    """
    extractor = KeywordExtractor()
    return extractor.extract_keywords(text, percentage)





# modify the below code to use on csv file

# Example usage
if __name__ == "__main__":
    
    # Extract keywords (15% of text length)
    keywords = extract_keywords_from_text(text)
    
    print(f" Text length: {len(text.split())} words")
    print(f" Extracted {len(keywords)} keywords (≈15% of text):")
    
    for i, keyword in enumerate(keywords, 1):
        print(f"{i:2d}. {keyword}")
