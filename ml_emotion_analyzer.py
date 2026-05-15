"""
Анализ эмоционального фона с использованием методов машинного обучения
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

class MLEmotionAnalyzer:    
    def __init__(self):
        print("Загрузка ML-моделей для анализа эмоционального фона...")
        
        self.sentence_model_available = False
        self.rubert_available = False
        self.bert_available = False
        
        try:
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.sentence_model_available = True
            print("✓ Загружена мультиязычная модель эмбеддингов")
        except Exception as e:
            print(f"✗ Не удалось загрузить sentence-transformers: {e}")
        
        try:
            self.rubert_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
            self.rubert_model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
            if torch.cuda.is_available():
                self.rubert_model = self.rubert_model.cuda()
            self.rubert_model.eval()
            self.rubert_available = True
            print("✓ Загружена модель RuBERT для русских текстов")
        except Exception as e:
            print(f"✗ Не удалось загрузить RuBERT: {e}")
        
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.cuda()
            self.bert_model.eval()
            self.bert_available = True
            print("✓ Загружена модель BERT для английских текстов")
        except Exception as e:
            print(f"✗ Не удалось загрузить BERT: {e}")
        
        self.model_available = self.sentence_model_available or self.rubert_available or self.bert_available
        print("✓ ML-анализатор эмоционального фона готов")
    
    def get_sentence_embedding(self, text, language='en'):
        if self.sentence_model_available:
            try:
                return self.sentence_model.encode(text)
            except:
                pass
        
        if language == 'en' and self.bert_available:
            try:
                inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                return embedding[0].cpu().numpy()
            except:
                pass
        
        if language == 'ru' and self.rubert_available:
            try:
                inputs = self.rubert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.rubert_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                return embedding[0].cpu().numpy()
            except:
                pass
        
        return None
    
    def calculate_ml_similarity(self, original_text, translated_text):
        if not self.model_available:
            return 0.5, "ML-модели недоступны, используется нейтральная оценка"
        
        orig_embedding = self.get_sentence_embedding(original_text, 'en')
        trans_embedding = self.get_sentence_embedding(translated_text, 'ru')
        
        if orig_embedding is None or trans_embedding is None:
            return 0.5, "Не удалось получить эмбеддинги, используется нейтральная оценка"
        
        try:
            similarity = cosine_similarity([orig_embedding], [trans_embedding])[0][0]
            similarity_normalized = (similarity + 1) / 2
            similarity_normalized = max(0.0, min(1.0, similarity_normalized))
            
            if similarity_normalized >= 0.8:
                interpretation = "ML-модели показывают высокое сходство эмоциональных профилей"
            elif similarity_normalized >= 0.6:
                interpretation = "ML-модели показывают среднее сходство, есть заметные расхождения"
            elif similarity_normalized >= 0.4:
                interpretation = "ML-модели показывают низкое сходство, эмоциональный фон различается"
            else:
                interpretation = "ML-модели показывают критическое расхождение, эмоции переданы противоположно"
            
            return similarity_normalized, interpretation
            
        except Exception as e:
            print(f"Ошибка вычисления сходства: {e}")
            return 0.5, "Ошибка вычисления ML-сходства"


class MLShiftDetector:
    def __init__(self):
        self.emotion_analyzer = MLEmotionAnalyzer()
        self.sentiment_available = False
        
        self.sentiment_mapping = {
            'POSITIVE': 'положительная',
            'NEGATIVE': 'отрицательная',
            'NEUTRAL': 'нейтральная',
            'positive': 'положительная',
            'negative': 'отрицательная',
            'neutral': 'нейтральная'
        }
        
        try:
            from transformers import pipeline
            self.en_sentiment = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.ru_sentiment = pipeline(
                "sentiment-analysis",
                model="blanchefort/rubert-base-cased-sentiment"
            )
            self.sentiment_available = True
            print("✓ Загружены модели для анализа тональности")
        except Exception as e:
            print(f"✗ Не удалось загрузить модели тональности: {e}")
    
    def _normalize_sentiment_label(self, label):
        return self.sentiment_mapping.get(label, label)
    
    def analyze_emotional_shift_ml(self, original_text, translated_text):
        result = {
            'original': {'label': 'не определена', 'score': 0.5, 'confidence': 0},
            'translated': {'label': 'не определена', 'score': 0.5, 'confidence': 0},
            'shift_detected': False,
            'shift_magnitude': 0,
            'interpretation': "",
            'ml_similarity': 0.5,
            'ml_interpretation': ""
        }
        
        if self.sentiment_available:
            try:
                en_result = self.en_sentiment(original_text[:512])[0]
                result['original']['label'] = self._normalize_sentiment_label(en_result['label'])
                result['original']['score'] = en_result['score']
                result['original']['confidence'] = en_result['score']
                
                ru_result = self.ru_sentiment(translated_text[:512])[0]
                result['translated']['label'] = self._normalize_sentiment_label(ru_result['label'])
                result['translated']['score'] = ru_result['score']
                result['translated']['confidence'] = ru_result['score']
            except Exception as e:
                print(f"Ошибка анализа тональности: {e}")
        
        ml_similarity, ml_interp = self.emotion_analyzer.calculate_ml_similarity(original_text, translated_text)
        result['ml_similarity'] = ml_similarity
        result['ml_interpretation'] = ml_interp
        
        if result['original']['label'] != 'не определена' and result['translated']['label'] != 'не определена':
            result['shift_detected'] = result['original']['label'] != result['translated']['label']
            result['shift_magnitude'] = abs(result['original']['score'] - result['translated']['score'])
            
            if result['shift_detected']:
                result['interpretation'] = f"В оригинале преобладает {result['original']['label']} тональность, а в переводе — {result['translated']['label']} тональность."
            else:
                if result['shift_magnitude'] > 0.3:
                    result['interpretation'] = f"Эмоциональная тональность сохранена ({result['original']['label']}), но интенсивность {'увеличилась' if result['translated']['score'] > result['original']['score'] else 'уменьшилась'}."
                else:
                    result['interpretation'] = f"Эмоциональная тональность сохранена хорошо. Преобладает {result['original']['label']} окраска."
        
        return result
    
    def calculate_final_emotion_score(self, lexical_score, ml_score):
        final_score = lexical_score * 0.5 + ml_score * 0.5
        
        if final_score >= 0.8:
            interpretation = "Отличная передача эмоционального фона"
        elif final_score >= 0.6:
            interpretation = "Хорошая передача эмоционального фона"
        elif final_score >= 0.4:
            interpretation = "Удовлетворительная передача эмоционального фона"
        else:
            interpretation = "Плохая передача эмоционального фона, требуется доработка"
        
        return {
            'final_score': round(final_score, 4),
            'final_percentage': round(final_score * 100, 1),
            'lexical_contribution': round(lexical_score * 0.5, 4),
            'ml_contribution': round(ml_score * 0.5, 4),
            'interpretation': interpretation
        }