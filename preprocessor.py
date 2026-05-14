"""
Предобработка текста и выравнивание предложений
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

class Preprocessor:
    def __init__(self):
        print("Загрузка моделей для предобработки...")
        self.model_available = False
        self.model = None
        self.tokenizer = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
            print("✓ Загружена модель для выравнивания: distiluse-base-multilingual-cased-v2")
            self.model_available = True
        except Exception as e:
            print(f"✗ Не удалось загрузить SentenceTransformer: {e}")
            self.model_available = False

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', '', text)
        return text.strip()

    def tokenize_sentences(self, text):
        sentences = re.split(r'[.!?;:]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_embedding(self, text):
        if not self.model_available:
            return None
        try:
            return self.sbert_model.encode(text)
        except Exception as e:
            print(f"Ошибка получения эмбеддинга: {e}")
            return None

    def align_sentences_simple(self, original_sentences, translated_sentences):
        aligned = []
        min_len = min(len(original_sentences), len(translated_sentences))
        
        for i in range(min_len):
            aligned.append((
                original_sentences[i],
                translated_sentences[i],
                1.0
            ))
        
        for i in range(min_len, len(original_sentences)):
            aligned.append((original_sentences[i], None, 0.0))
        
        return aligned

    def align_sentences_semantic(self, original_sentences, translated_sentences):
        if not self.model_available or len(original_sentences) == 0 or len(translated_sentences) == 0:
            return self.align_sentences_simple(original_sentences, translated_sentences)
        
        orig_embeds = []
        trans_embeds = []
        
        for sent in original_sentences:
            emb = self.get_embedding(sent)
            if emb is not None:
                orig_embeds.append(emb)
            else:
                orig_embeds.append(np.random.randn(512))
        
        for sent in translated_sentences:
            emb = self.get_embedding(sent)
            if emb is not None:
                trans_embeds.append(emb)
            else:
                trans_embeds.append(np.random.randn(512))
        
        if len(orig_embeds) == 0 or len(trans_embeds) == 0:
            return self.align_sentences_simple(original_sentences, translated_sentences)
        
        orig_embeds = np.array(orig_embeds)
        trans_embeds = np.array(trans_embeds)
        
        similarity = cosine_similarity(orig_embeds, trans_embeds)
        
        aligned = []
        used_trans = set()
        
        for i, orig_sent in enumerate(original_sentences):
            best_idx = -1
            best_score = 0
            
            search_range = range(max(0, i-3), min(len(translated_sentences), i+4))
            
            for j in search_range:
                if j not in used_trans and j < len(similarity[i]):
                    score = float(similarity[i, j])
                    if score > best_score and score > 0.4:
                        best_score = score
                        best_idx = j
            
            if best_idx >= 0:
                aligned.append((original_sentences[i], translated_sentences[best_idx], best_score))
                used_trans.add(best_idx)
            else:
                aligned.append((original_sentences[i], None, 0.0))
        
        return aligned

    def preprocess_pair(self, original, translated):
        orig_clean = self.clean_text(original)
        trans_clean = self.clean_text(translated)

        orig_sentences = self.tokenize_sentences(orig_clean)
        trans_sentences = self.tokenize_sentences(trans_clean)

        aligned = self.align_sentences_semantic(orig_sentences, trans_sentences)

        translation_map = {}
        for orig, trans, score in aligned:
            if trans is not None:
                translation_map[orig.lower().strip()] = trans

        return {
            'original_clean': orig_clean,
            'translated_clean': trans_clean,
            'original_sentences': orig_sentences,
            'translated_sentences': trans_sentences,
            'aligned': aligned,
            'translation_map': translation_map
        }

    def get_model_available(self):
        return self.model_available