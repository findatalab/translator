import re
import math
import csv

def load_cultural_markers(csv_path="cultural_markers_merged.csv"):
    cultural_markers = set()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) > 0:
                    marker = row[0].strip().lower()
                    if marker:
                        cultural_markers.add(marker)
        print(f"Загружено {len(cultural_markers)} культурных маркеров")
    except FileNotFoundError:
        print(f"Файл {csv_path} не найден")
    except Exception as e:
        print(f"Ошибка загрузки культурных маркеров: {e}")
    
    return cultural_markers

CULTURAL_MARKERS = load_cultural_markers()

positive_words = {
    "good","great","happy","joy","love","hope","smile","wonderful","kind",
    "счастье","радость","любовь","надежда","улыбка","прекрасно","хорошо",
    "excellent","amazing","fantastic","beautiful","glad","pleased","proud"
}

negative_words = {
    "bad","sad","pain","hate","fear","anger","terrible","cry",
    "грусть","боль","ненависть","страх","злость","ужасно","плохо",
    "awful","horrible","depressed","angry","afraid","scared","suffer"
}

intensifiers = {
    "very","extremely","so","really","too","absolutely","quite","rather",
    "очень","крайне","слишком","совсем","ужасно","невероятно"
}

def find_cultural_markers(text):
    if not CULTURAL_MARKERS:
        return []
    
    text_lower = text.lower()
    found_markers = []
    
    for marker in CULTURAL_MARKERS:
        if marker in text_lower:
            if len(marker) < 4:
                pattern = r'\b' + re.escape(marker) + r'\b'
                if re.search(pattern, text_lower):
                    found_markers.append(marker)
            else:
                found_markers.append(marker)
    
    return list(set(found_markers))

def assess_translation_risk(cultural_markers, original_metrics, translation_metrics):
    if not cultural_markers:
        return {
            "has_cultural_markers": False,
            "risk_level": "low",
            "recommendation": "Культурные маркеры не обнаружены"
        }
    
    polarity_shift = abs(original_metrics["polarity"] - translation_metrics["polarity"])
    density_shift = abs(original_metrics["density"] - translation_metrics["density"])
    
    if polarity_shift > 0.3 or density_shift > 0.2:
        risk_level = "high"
        recommendation = "Внимание: Обнаружены культурные маркеры, которые могут исказить эмоциональный фон при переводе"
    elif polarity_shift > 0.15 or density_shift > 0.1:
        risk_level = "medium"
        recommendation = "Осторожно: Культурные маркеры могут частично изменить эмоциональную окраску"
    else:
        risk_level = "low"
        recommendation = "Культурные маркеры не должны существенно повлиять на эмоциональный фон"
    
    return {
        "has_cultural_markers": True,
        "markers_found": cultural_markers,
        "count": len(cultural_markers),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "polarity_shift": round(polarity_shift, 3),
        "density_shift": round(density_shift, 3)
    }

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def analyze_text(text):
    words = tokenize(text)
    
    pos = 0
    neg = 0
    intensity = 0
    
    for w in words:
        if w in positive_words:
            pos += 1
        if w in negative_words:
            neg += 1
        if w in intensifiers:
            intensity += 1
    
    emotional_words = pos + neg
    
    density = emotional_words / len(words) if words else 0
    polarity = (pos - neg) / emotional_words if emotional_words else 0
    
    if pos > neg:
        dominant = "positive"
    elif neg > pos:
        dominant = "negative"
    else:
        dominant = "neutral"
    
    balance = pos / emotional_words if emotional_words else 0
    
    cultural_markers = find_cultural_markers(text)
    
    return {
        "word_count": len(words),
        "positive": pos,
        "negative": neg,
        "intensity": intensity,
        "density": round(density, 3),
        "polarity": round(polarity, 3),
        "balance": round(balance, 3),
        "dominant": dominant,
        "cultural_markers": cultural_markers,
        "cultural_markers_count": len(cultural_markers)
    }

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    m1 = math.sqrt(sum(a * a for a in v1))
    m2 = math.sqrt(sum(b * b for b in v2))
    
    if m1 == 0 or m2 == 0:
        return 0
    
    return dot / (m1 * m2)

def compare_emotions(m1, m2):
    vector1 = [
        m1["polarity"],
        m1["density"],
        m1["balance"],
        m1["positive"],
        m1["negative"],
        m1["intensity"]
    ]
    
    vector2 = [
        m2["polarity"],
        m2["density"],
        m2["balance"],
        m2["positive"],
        m2["negative"],
        m2["intensity"]
    ]
    
    similarity = cosine_similarity(vector1, vector2) * 100
    
    polarity_diff = abs(m1["polarity"] - m2["polarity"])
    density_diff = abs(m1["density"] - m2["density"])
    
    emotion_match = m1["dominant"] == m2["dominant"]
    
    preservation = max(0, 100 - (polarity_diff * 100 + density_diff * 50))
    
    risk_assessment = assess_translation_risk(
        m1["cultural_markers"], m1, m2
    )
    
    return {
        "similarity": round(similarity, 2),
        "polarity_difference": round(polarity_diff, 3),
        "density_difference": round(density_diff, 3),
        "emotion_match": emotion_match,
        "emotion_preservation": round(preservation, 2),
        "cultural_markers_original": m1["cultural_markers"],
        "cultural_markers_count_original": m1["cultural_markers_count"],
        "translation_risk": risk_assessment
    }