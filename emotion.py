import re
import math

positive_words = {
    "good","great","happy","joy","love","hope","smile","wonderful","kind",
    "счастье","радость","любовь","надежда","улыбка","прекрасно","хорошо"
}

negative_words = {
    "bad","sad","pain","hate","fear","anger","terrible","cry",
    "грусть","боль","ненависть","страх","злость","ужасно","плохо"
}

intensifiers = {
    "very","extremely","so","really","too","absolutely",
    "очень","крайне","слишком","совсем"
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

    return {
        "word_count": len(words),
        "positive": pos,
        "negative": neg,
        "intensity": intensity,
        "density": round(density, 3),
        "polarity": round(polarity, 3),
        "balance": round(balance, 3),
        "dominant": dominant
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

    return {
        "similarity": round(similarity, 2),
        "polarity_difference": round(polarity_diff, 3),
        "density_difference": round(density_diff, 3),
        "emotion_match": emotion_match,
        "emotion_preservation": round(preservation, 2)
    }
