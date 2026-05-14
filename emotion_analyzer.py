"""
Анализ эмоционального профиля по 8 эмоциям Плутчика
"""

class EmotionalProfileAnalyzer:
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.plutchik_emotions = lexicon.plutchik_emotions
        self.russian_emotions = {
            'anger': 'гнев',
            'anticipation': 'ожидание',
            'disgust': 'отвращение',
            'fear': 'страх',
            'joy': 'радость',
            'sadness': 'печаль',
            'surprise': 'удивление',
            'trust': 'доверие'
        }

    def analyze(self, text, language='english'):
        emotion_vector, emotional_words = self.lexicon.get_emotion_vector(text, language)

        total = sum(emotion_vector.values())
        percentages = {e: round((emotion_vector[e] / total) * 100, 2) if total > 0 else 0
                       for e in emotion_vector}

        dominant = max(emotion_vector, key=emotion_vector.get) if max(emotion_vector.values()) > 0 else None

        interpretation = self._get_interpretation(percentages, dominant)

        return {
            'emotions': emotion_vector,
            'percentages': percentages,
            'dominant': dominant,
            'dominant_russian': self.russian_emotions.get(dominant, 'неопределена') if dominant else 'нейтральная',
            'dominant_intensity': max(emotion_vector.values()) if max(emotion_vector.values()) > 0 else 0,
            'emotional_words_count': emotional_words,
            'interpretation': interpretation
        }

    def _get_interpretation(self, percentages, dominant):
        if not dominant:
            return "Текст имеет нейтральный эмоциональный фон."

        if percentages.get(dominant, 0) > 50:
            intensity = "доминирует"
        elif percentages.get(dominant, 0) > 30:
            intensity = "ярко выражена"
        else:
            intensity = "присутствует"

        return f"В тексте {intensity} эмоция '{self.russian_emotions[dominant]}'. Интенсивность: {percentages.get(dominant, 0)}%"