"""
Анализ VAD-профиля (Valence, Arousal, Dominance)
"""

import math

class VADProfileAnalyzer:
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def analyze(self, text):
        vad_result = self.lexicon.get_vad_vector(text)

        vad_result['valence_interpretation'] = self._interpret_valence(vad_result['valence'])
        vad_result['arousal_interpretation'] = self._interpret_arousal(vad_result['arousal'])
        vad_result['dominance_interpretation'] = self._interpret_dominance(vad_result['dominance'])
        vad_result['overall_interpretation'] = self._overall_interpretation(vad_result)

        return vad_result

    def _interpret_valence(self, valence):
        if valence >= 0.7:
            return "Сильно положительная: оптимизм, радость"
        if valence >= 0.55:
            return "Положительная: преобладает позитив"
        if valence >= 0.45:
            return "Нейтральная: сбалансированный текст"
        if valence >= 0.3:
            return "Отрицательная: преобладает негатив"
        return "Сильно отрицательная: глубокий негатив"

    def _interpret_arousal(self, arousal):
        if arousal >= 0.7:
            return "Высокая активность: интенсивный, драматичный"
        if arousal >= 0.55:
            return "Повышенная активность: динамичный"
        if arousal >= 0.45:
            return "Умеренная активность: сбалансированный"
        if arousal >= 0.3:
            return "Пониженная активность: спокойный"
        return "Низкая активность: минимальная интенсивность"

    def _interpret_dominance(self, dominance):
        if dominance >= 0.7:
            return "Высокая доминантность: уверенный тон"
        if dominance >= 0.55:
            return "Уверенное повествование"
        if dominance >= 0.45:
            return "Нейтральная позиция"
        if dominance >= 0.3:
            return "Неуверенный тон"
        return "Низкая доминантность: беспомощность"

    def _overall_interpretation(self, vad):
        if vad['coverage'] < 0.3:
            return f"✗ Словарь покрывает {int(vad['coverage'] * 100)}% слов"

        valence_desc = "положительная" if vad['valence'] >= 0.5 else "отрицательная" if vad['valence'] < 0.5 else "нейтральная"
        arousal_desc = "высокой" if vad['arousal'] >= 0.55 else "низкой" if vad['arousal'] < 0.45 else "умеренной"

        return f"Текст характеризуется {valence_desc} валентностью, {arousal_desc} активностью"

    def calculate_distance(self, vad_orig, vad_trans):
        distance = math.sqrt(
            (vad_orig['valence'] - vad_trans['valence']) ** 2 +
            (vad_orig['arousal'] - vad_trans['arousal']) ** 2 +
            (vad_orig['dominance'] - vad_trans['dominance']) ** 2
        )
        normalized = distance / 1.732

        if normalized < 0.2:
            interpretation = "Отличное совпадение"
        elif normalized < 0.4:
            interpretation = "Хорошее совпадение"
        elif normalized < 0.6:
            interpretation = "Среднее совпадение"
        else:
            interpretation = "Сильное расхождение"

        return {
            'distance': round(distance, 4),
            'normalized_distance': round(normalized, 4),
            'interpretation': interpretation
        }