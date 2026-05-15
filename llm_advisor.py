"""
LLM-советник для анализа эмоциональной окраски перевода
Использует Ollama с моделью qwen2.5:7b
"""

import requests
import json
import re
import time
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

class LLMAdvisor:
    
    def __init__(self, ollama_model="qwen2.5:7b"):
        self.ollama_model = ollama_model
        self.ollama_available = False
        self.ollama_url = "http://localhost:11434/api/generate"
        
        self.model_available = False
        self.sentiment_available = False
        self.en_sentiment_available = False
        
        print(f"Проверка Ollama с моделью {ollama_model}...")
        
        try:
            tags_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if tags_response.status_code == 200:
                installed_models = tags_response.json().get("models", [])
                installed_names = [m.get("name", "") for m in installed_models]
                
                model_found = False
                for m in installed_names:
                    if self.ollama_model in m:
                        model_found = True
                        break
                
                if model_found:
                    print(f"✓ Модель {self.ollama_model} найдена")
                    self.ollama_available = True
                    self.model_available = True
                else:
                    print(f"✗ Модель {self.ollama_model} не найдена")
                    print(f"   Доступные модели: {installed_names}")
        except Exception as e:
            print(f"✗ Ollama не доступна: {e}")
        
        print("Загрузка моделей для анализа настроений...")
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="blanchefort/rubert-base-cased-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            self.sentiment_available = True
            print("✓ Загружена модель анализа настроений (rubert-base-cased-sentiment)")
        except Exception as e:
            print(f"✗ Не удалось загрузить модель настроений: {e}")
        
        try:
            self.en_sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            self.en_sentiment_available = True
            print("✓ Загружена английская модель анализа настроений")
        except Exception as e:
            print(f"✗ Не удалось загрузить английскую модель: {e}")
        
        print("✓ LLM-советник готов к работе")
    
    def _call_ollama(self, prompt, max_tokens=600):
        if not self.ollama_available:
            return None
        
        for attempt in range(2):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5,
                            "num_predict": max_tokens,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1
                        }
                    },
                    timeout=180
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    print(f"Ollama ошибка (попытка {attempt+1}): {response.status_code}")
                    time.sleep(3)
                    
            except requests.exceptions.Timeout:
                print(f"Ollama таймаут (попытка {attempt+1})")
                time.sleep(3)
            except Exception as e:
                print(f"Ошибка Ollama (попытка {attempt+1}): {e}")
                time.sleep(3)
        
        return None
    
    def generate_recommendation(self, original_text, translated_text, eqa_score, 
                                 ed_score, pde_score, k_score, dominant_orig, dominant_trans):
        if self.ollama_available:
            result = self._generate_ollama_recommendation(
                original_text, translated_text, eqa_score,
                ed_score, pde_score, k_score, dominant_orig, dominant_trans
            )
            if result and len(result) > 50:
                return result
        
        return self._fallback_recommendation(eqa_score, ed_score, pde_score, k_score, 
                                             dominant_orig, dominant_trans)
    
    def _generate_ollama_recommendation(self, original_text, translated_text, eqa_score, 
                                         ed_score, pde_score, k_score, dominant_orig, dominant_trans):
        if len(original_text) > 500:
            orig_short = original_text[:250] + "\n...\n" + original_text[-250:]
            trans_short = translated_text[:250] + "\n...\n" + translated_text[-250:]
        else:
            orig_short = original_text
            trans_short = translated_text
        
        if eqa_score >= 0.8:
            eqa_text = "отличное качество перевода, эмоциональный окрас сохранён превосходно"
        elif eqa_score >= 0.6:
            eqa_text = "хорошее качество перевода, но есть небольшие эмоциональные расхождения"
        elif eqa_score >= 0.4:
            eqa_text = "удовлетворительное качество, заметные эмоциональные искажения"
        else:
            eqa_text = "низкое качество, серьёзные эмоциональные потери"
        
        if ed_score < 0.2:
            ed_text = "эмоциональный тон совпадает почти идеально"
        elif ed_score < 0.4:
            ed_text = "эмоциональный тон близок к оригиналу"
        elif ed_score < 0.6:
            ed_text = "эмоциональный тон заметно отличается"
        else:
            ed_text = "эмоциональный тон сильно искажён"
        
        if pde_score >= 0.8:
            pde_text = f"доминирующая эмоция '{dominant_orig}' сохранена полностью"
        elif pde_score >= 0.6:
            pde_text = f"доминирующая эмоция '{dominant_orig}' сохранена, но ослаблена"
        elif pde_score >= 0.4:
            pde_text = f"доминирующая эмоция сместилась с '{dominant_orig}' на '{dominant_trans}'"
        else:
            pde_text = f"доминирующая эмоция утеряна, оригинальная '{dominant_orig}' заменена на '{dominant_trans}'"
        
        if k_score >= 0.8:
            k_text = "культурные маркеры адаптированы отлично, эмоциональный подтекст сохранён"
        elif k_score >= 0.6:
            k_text = "культурные маркеры адаптированы хорошо, но есть небольшие потери"
        elif k_score >= 0.4:
            k_text = "культурные маркеры адаптированы удовлетворительно, заметны потери смысла"
        else:
            k_text = "культурные маркеры практически не адаптированы, эмоциональный подтекст утерян"
        
        prompt = f"""Ты эксперт по межкультурной коммуникации и художественному переводу. Проанализируй перевод и дай конкретные рекомендации, учитывая культурные различия между англоязычной и русскоязычной аудиториями.

ОРИГИНАЛ (английский):
{orig_short}

ПЕРЕВОД (русский):
{trans_short}

ОБЩАЯ ОЦЕНКА КАЧЕСТВА:
{eqa_text} (EQA = {eqa_score:.2f})

ДЕТАЛЬНЫЕ МЕТРИКИ:
1) Эмоциональный тон: {ed_text} (ED = {ed_score:.2f}, где 0 - идеально, 1 - полное расхождение)
2) Сохранение главной эмоции: {pde_text} (PDE = {pde_score:.2f})
3) Адаптация культурных маркеров: {k_text} (K = {k_score:.2f})
4) Доминанта оригинала: {dominant_orig}
5) Доминанта перевода: {dominant_trans}

Напиши развёрнутые рекомендации переводчику. Обрати особое внимание на:
1. Как лучше передать эмоциональную окраску с учётом культурных особенностей русскоязычного читателя
2. Конкретные примеры замен для ключевых эмоционально окрашенных фрагментов
3. Стратегии адаптации культурных маркеров, которые не теряют эмоциональный подтекст

Пиши в виде связного текста из 4-6 предложений. Не используй маркированные списки. Будь конкретна.

РЕКОМЕНДАЦИИ:"""
        
        result = self._call_ollama(prompt, max_tokens=600)
        
        if result:
            result = re.sub(r'^РЕКОМЕНДАЦИИ:\s*', '', result)
            result = re.sub(r'^Рекомендации:\s*', '', result)
        
        return result
    
    def generate_marker_recommendation(self, marker, original_sentence, translated_sentence, loss):
        if self.ollama_available:
            result = self._generate_ollama_marker_recommendation(
                marker, original_sentence, translated_sentence, loss
            )
            if result and len(result) > 30:
                return result
        
        return self._fallback_marker_recommendation(marker, loss)
    
    def _generate_ollama_marker_recommendation(self, marker, original_sentence, translated_sentence, loss):
        orig_short = original_sentence[:300] + "..." if len(original_sentence) > 300 else original_sentence
        trans_short = translated_sentence[:300] + "..." if len(translated_sentence) > 300 else translated_sentence
        
        if loss > 0.8:
            severity = "критическая потеря эмоционального фона"
        elif loss > 0.6:
            severity = "большая потеря эмоционального фона"
        elif loss > 0.4:
            severity = "средняя потеря эмоционального фона"
        else:
            severity = "незначительная потеря эмоционального фона"
        
        prompt = f"""Ты эксперт по переводу культурно-специфической лексики с английского на русский.

КУЛЬТУРНЫЙ МАРКЕР: "{marker}"
Степень проблемы: {severity} (потеря {loss:.0%})

ОРИГИНАЛ:
{orig_short}

СУЩЕСТВУЮЩИЙ ПЕРЕВОД:
{trans_short}

Дай развёрнутую рекомендацию переводчику из 3-4 предложений. Объясни, почему текущий перевод не передаёт эмоциональный подтекст, и предложи конкретный альтернативный вариант перевода с сохранением культурного и эмоционального смысла. Учитывай различия между английской и русской культурой.

РЕКОМЕНДАЦИЯ:"""
        
        result = self._call_ollama(prompt, max_tokens=350)
        
        if result:
            result = re.sub(r'^РЕКОМЕНДАЦИЯ:\s*', '', result)
        
        return result
    
    def analyze_emotional_shift(self, original_text, translated_text):
        result = {
            'original': {'label': 'не определена', 'score': 0.5},
            'translated': {'label': 'не определена', 'score': 0.5},
            'shift_detected': False,
            'shift_magnitude': 0,
            'interpretation': "Анализ выполняется на основе метрик системы"
        }
        
        if self.en_sentiment_available:
            try:
                orig_result = self.en_sentiment_pipeline(original_text[:512])[0]
                label_map = {'POSITIVE': 'положительная', 'NEGATIVE': 'отрицательная', 'NEUTRAL': 'нейтральная'}
                result['original']['label'] = label_map.get(orig_result['label'], orig_result['label'])
                result['original']['score'] = orig_result['score']
            except:
                pass
        
        if self.sentiment_available:
            try:
                trans_result = self.sentiment_pipeline(translated_text[:512])[0]
                label_map_ru = {'positive': 'положительная', 'negative': 'отрицательная', 'neutral': 'нейтральная'}
                result['translated']['label'] = label_map_ru.get(trans_result['label'], trans_result['label'])
                result['translated']['score'] = trans_result['score']
            except:
                pass
        
        if result['original']['label'] != 'не определена' and result['translated']['label'] != 'не определена':
            result['shift_detected'] = result['original']['label'] != result['translated']['label']
            if result['shift_detected']:
                result['interpretation'] = f"В оригинале преобладает {result['original']['label']} тональность, а в переводе — {result['translated']['label']} тональность."
            else:
                result['interpretation'] = f"Эмоциональная тональность сохранена ({result['original']['label']})."
        
        return result
    
    def _fallback_recommendation(self, eqa_score, ed_score, pde_score, k_score, dominant_orig, dominant_trans):
        if eqa_score >= 0.8:
            return f"Перевод выполнен на высоком уровне. Эмоциональный окрас сохранён отлично. {dominant_orig} передана естественно. Культурные маркеры адаптированы хорошо. Рекомендуется проверить только единичные случаи культурно-специфической лексики."
        
        elif eqa_score >= 0.6:
            text = f"Перевод в целом хороший, но требует небольших доработок. "
            if ed_score > 0.35:
                text += f"Общая эмоциональная тональность немного отличается от оригинала. "
            if pde_score < 0.7:
                text += f"Доминирующая эмоция '{dominant_orig}' передана не в полной силе. Усильте её с помощью более экспрессивной лексики. "
            if k_score < 0.6:
                text += f"Некоторые культурные маркеры требуют лучшей адаптации. Вместо буквального перевода найдите русские аналоги. "
            return text
        
        elif eqa_score >= 0.4:
            text = f"Перевод требует доработки. Эмоциональная окраска передана неточно. "
            if ed_score > 0.5:
                text += f"Общая тональность перевода сильно отличается от оригинала. "
            if pde_score < 0.5:
                text += f"Доминирующая эмоция изменилась с '{dominant_orig}' на '{dominant_trans}'. Верните акцент на исходную эмоцию. "
            if k_score < 0.4:
                text += f"Культурные маркеры практически не сохранены. Используйте функциональную замену. "
            return text
        
        else:
            return f"Перевод нуждается в серьёзной доработке. Эмоциональный профиль значительно отличается от оригинала. Рекомендуется полностью сверить перевод с оригиналом и восстановить эмоциональные акценты."
    
    def _fallback_marker_recommendation(self, marker, loss):
        if loss > 0.8:
            return f"Маркер '{marker}' практически полностью потерял эмоциональную окраску. Рекомендуется найти русский аналог, который вызывает у читателя ту же эмоциональную реакцию, что и оригинал. Используйте функциональную замену вместо буквального перевода."
        elif loss > 0.6:
            return f"Маркер '{marker}' передан с заметными потерями. Попробуйте заменить буквальный перевод на более естественный для русского языка вариант, сохраняющий идиоматичность и эмоциональный оттенок оригинала."
        elif loss > 0.4:
            return f"Маркер '{marker}' требует уточнения. Смысл передан верно, но эмоциональный оттенок частично утерян. Добавьте эмоционально окрашенные слова или измените порядок слов для усиления эффекта."
        else:
            return f"Маркер '{marker}' передан хорошо. Эмоциональная окраска сохранена. Для улучшения можно проверить естественность звучания фразы в контексте всего абзаца."