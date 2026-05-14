"""
Расчёт метрик EQA: ED, PDE, K
"""

import math
import numpy as np

class EQAMetricsCalculator:
    def __init__(self, vad_analyzer, emotion_analyzer, preprocessor):
        self.vad_analyzer = vad_analyzer
        self.emotion_analyzer = emotion_analyzer
        self.preprocessor = preprocessor
        self.weights = {'ed': 0.5, 'pde': 0.3, 'k': 0.2}
        self.thresholds = {'excellent': 0.8, 'good': 0.6, 'satisfactory': 0.4}

    def calculate_ed(self, vad_orig, vad_trans):
        distance = math.sqrt(
            (vad_orig['valence'] - vad_trans['valence']) ** 2 +
            (vad_orig['arousal'] - vad_trans['arousal']) ** 2 +
            (vad_orig['dominance'] - vad_trans['dominance']) ** 2
        )
        normalized = distance / 1.732

        if normalized < 0.2:
            interpretation = "VAD-профили практически идентичны"
        elif normalized < 0.4:
            interpretation = "Эмоциональная окраска сохранена хорошо"
        elif normalized < 0.6:
            interpretation = "Есть заметные различия в эмоциональной окраске"
        else:
            interpretation = "Эмоциональный тон передан с серьёзными искажениями"

        return {
            'normalized_distance': round(normalized, 4),
            'interpretation': interpretation,
            'weight': self.weights['ed']
        }

    def calculate_pde(self, emotion_orig, emotion_trans):
        dom_orig = emotion_orig['dominant']
        dom_trans = emotion_trans['dominant']

        base_match = 1.0 if dom_orig == dom_trans and dom_orig is not None else 0.0

        if dom_orig and dom_orig in emotion_trans['emotions']:
            intensity_diff = abs(emotion_orig['emotions'][dom_orig] - emotion_trans['emotions'][dom_orig])
            intensity_factor = 1.0 - min(intensity_diff, 1.0)
        else:
            intensity_factor = 0.0

        pde = base_match * 0.7 + intensity_factor * 0.3

        if pde >= 0.9:
            interpretation = "Доминирующая эмоция полностью сохранена"
        elif pde >= 0.7:
            interpretation = "Доминирующая эмоция сохранена хорошо"
        elif pde >= 0.5:
            interpretation = "Доминирующая эмоция частично сохранена"
        else:
            interpretation = "Доминирующая эмоция утеряна"

        return {
            'pde_score': round(pde, 4),
            'interpretation': interpretation,
            'dominant_match': base_match == 1.0,
            'weight': self.weights['pde']
        }

    def calculate_k(self, original_sentences, translated_sentences, aligned_pairs, cultural_markers):
        if not cultural_markers:
            return {
                'k_score': 1.0,
                'interpretation': "Культурные маркеры не обнаружены",
                'problematic_markers': [],
                'weight': self.weights['k']
            }

        translation_map = {}
        for orig, trans, score in aligned_pairs:
            if trans is not None:
                normalized_orig = orig.lower().strip()
                translation_map[normalized_orig] = trans

        sentence_scores = []
        seen_markers = set()
        
        marker_to_sentence = {}
        for sent in original_sentences:
            sent_lower = sent.lower()
            for marker_info in cultural_markers:
                marker = marker_info['marker']
                marker_lower = marker.lower()
                if marker_lower in sent_lower:
                    if marker not in marker_to_sentence:
                        marker_to_sentence[marker] = sent

        for marker_info in cultural_markers[:20]:
            marker = marker_info['marker']
            
            if marker in seen_markers:
                continue
            seen_markers.add(marker)
            
            orig_sentence = marker_to_sentence.get(marker)
            if not orig_sentence:
                continue
            
            trans_sentence = translation_map.get(orig_sentence.lower().strip())
            
            if trans_sentence:
                orig_emotion = self.emotion_analyzer.analyze(orig_sentence, 'english')
                trans_emotion = self.emotion_analyzer.analyze(trans_sentence, 'russian')
                
                orig_vec = [orig_emotion['emotions'][e] for e in self.emotion_analyzer.plutchik_emotions]
                trans_vec = [trans_emotion['emotions'][e] for e in self.emotion_analyzer.plutchik_emotions]
                
                similarity = self._cosine_similarity(orig_vec, trans_vec)
                loss = 1 - similarity
                
                sentence_scores.append({
                    'marker': marker,
                    'original_sentence': orig_sentence,
                    'translated_sentence': trans_sentence,
                    'similarity': round(similarity, 4),
                    'loss': round(loss, 4),
                    'is_problematic': loss > 0.8
                })

        if sentence_scores:
            avg_loss = np.mean([s['loss'] for s in sentence_scores])
            k_score = max(0, 1 - avg_loss)
        else:
            k_score = 1.0

        if k_score >= self.thresholds['excellent']:
            interpretation = "Отличная культурная адаптация. Эмоциональный фон сохранён."
        elif k_score >= self.thresholds['good']:
            interpretation = "Хорошая культурная адаптация. Небольшие потери."
        elif k_score >= self.thresholds['satisfactory']:
            interpretation = "Удовлетворительная адаптация. Заметные потери."
        else:
            interpretation = "Плохая культурная адаптация. Значительные искажения."

        return {
            'k_score': round(k_score, 4),
            'interpretation': interpretation,
            'problematic_markers': [s for s in sentence_scores if s['is_problematic']],
            'weight': self.weights['k']
        }

    def _cosine_similarity(self, a, b):
        if not a or not b or sum(a) == 0 or sum(b) == 0:
            return 0.5
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    def calculate_eqa(self, ed_result, pde_result, k_result):
        ed_similarity = 1 - ed_result['normalized_distance']

        eqa = (ed_similarity * 0.5 +
               pde_result['pde_score'] * 0.3 +
               k_result['k_score'] * 0.2)
        eqa = round(eqa, 4)

        if eqa >= 0.8:
            quality = "Отлично"
            interpretation = "Высокое качество перевода! Эмоциональный окрас сохранён превосходно."
        elif eqa >= 0.6:
            quality = "Хорошо"
            interpretation = "Хорошее качество перевода. Основные эмоциональные акценты переданы правильно."
        elif eqa >= 0.4:
            quality = "Удовлетворительно"
            interpretation = "Удовлетворительное качество. Эмоциональный окрас передан с заметными искажениями."
        else:
            quality = "Низкое"
            interpretation = "Низкое качество перевода. Требуется полный пересмотр перевода."

        return {
            'eqa_score': eqa,
            'quality': quality,
            'interpretation': interpretation
        }