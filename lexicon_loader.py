"""
Загрузка и управление лексиконами
"""

import pandas as pd
import re

class LexiconLoader:
    def __init__(self):
        self.plutchik_emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
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

        self.english_emotion_lexicon = {}
        self.russian_emotion_lexicon = {}
        self.vad_lexicon = {}
        self.cultural_markers = []

        self.load_emotion_lexicons()
        self.load_vad_lexicon()
        self.load_cultural_markers()

    def tokenize_words(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def load_emotion_lexicons(self):
        try:
            with open("data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        word = parts[0].lower()
                        emotion = parts[1]
                        value = int(parts[2])

                        if word not in self.english_emotion_lexicon:
                            self.english_emotion_lexicon[word] = {e: 0 for e in self.plutchik_emotions}

                        if emotion in self.plutchik_emotions:
                            self.english_emotion_lexicon[word][emotion] = value
            print(f"✓ Английский словарь эмоций: {len(self.english_emotion_lexicon)} слов")
        except Exception as e:
            print(f"✗ Ошибка загрузки английского лексикона: {e}")

        try:
            with open("data/Russian-NRC-EmoLex.txt", 'r', encoding='utf-8') as f:
                header = f.readline().strip().split('\t')
                emotion_cols = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
                
                for line_num, line in enumerate(f, 2):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 12:
                        russian_word = parts[-1].lower().strip()
                        if russian_word and russian_word != 'nan':
                            emotion_vector = {}
                            for i, emotion in enumerate(emotion_cols, 1):
                                if i < len(parts):
                                    try:
                                        emotion_vector[emotion] = int(float(parts[i]))
                                    except (ValueError, IndexError):
                                        emotion_vector[emotion] = 0
                                else:
                                    emotion_vector[emotion] = 0
                            
                            self.russian_emotion_lexicon[russian_word] = emotion_vector
            
            print(f"✓ Русский словарь эмоций: {len(self.russian_emotion_lexicon)} слов")
        except FileNotFoundError:
            print(f"✗ Файл Russian-NRC-EmoLex.txt не найден в папке data/")
            self.russian_emotion_lexicon = {}
        except Exception as e:
            print(f"✗ Ошибка загрузки русского лексикона: {e}")
            self.russian_emotion_lexicon = {}

    def load_vad_lexicon(self):
        try:
            with open("data/NRC-VAD-Lexicon.txt", 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    english_word = parts[0].lower()
                    try:
                        valence = float(parts[1])
                        arousal = float(parts[2])
                        dominance = float(parts[3])
                        russian_word = parts[4].lower()

                        vad_data = {'valence': valence, 'arousal': arousal, 'dominance': dominance}
                        self.vad_lexicon[english_word] = vad_data
                        self.vad_lexicon[russian_word] = vad_data
                    except ValueError:
                        continue

            print(f"✓ VAD словарь: {len(self.vad_lexicon)} слов")
        except Exception as e:
            print(f"✗ Ошибка загрузки VAD: {e}")

    def load_cultural_markers(self):
        try:
            df = pd.read_csv("data/cultural_markers.csv", encoding='utf-8')
            if 'cultural_marker' in df.columns:
                self.cultural_markers = [str(m).lower().strip() for m in df['cultural_marker'].dropna()]
            else:
                self.cultural_markers = [str(m).lower().strip() for m in df.iloc[:, 0].dropna()]
            print(f"✓ Культурные маркеры: {len(self.cultural_markers)}")
        except Exception as e:
            print(f"✗ Ошибка загрузки маркеров: {e}")
            self.cultural_markers = []

    def get_emotion_vector(self, text, language='english'):
        words = self.tokenize_words(text)
        
        if language == 'english':
            lexicon = self.english_emotion_lexicon
        else:
            lexicon = self.russian_emotion_lexicon if self.russian_emotion_lexicon else self.english_emotion_lexicon

        emotion_counts = {e: 0 for e in self.plutchik_emotions}
        emotional_words = 0

        for word in words:
            if word in lexicon:
                for emotion in self.plutchik_emotions:
                    emotion_counts[emotion] += lexicon[word].get(emotion, 0)
                emotional_words += 1

        if emotional_words > 0:
            for e in emotion_counts:
                emotion_counts[e] = round(emotion_counts[e] / emotional_words, 4)

        return emotion_counts, emotional_words

    def get_vad_vector(self, text):
        words = self.tokenize_words(text)

        valence_sum = arousal_sum = dominance_sum = 0
        words_found = 0

        for word in words:
            if word in self.vad_lexicon:
                valence_sum += self.vad_lexicon[word]['valence']
                arousal_sum += self.vad_lexicon[word]['arousal']
                dominance_sum += self.vad_lexicon[word]['dominance']
                words_found += 1

        if words_found > 0:
            return {
                'valence': round(valence_sum / words_found, 4),
                'arousal': round(arousal_sum / words_found, 4),
                'dominance': round(dominance_sum / words_found, 4),
                'words_found': words_found,
                'total_words': len(words),
                'coverage': round(words_found / len(words), 3) if words else 0
            }
        else:
            return {
                'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5,
                'words_found': 0, 'total_words': len(words), 'coverage': 0
            }