"""
Приложение для анализа эмоционального окраса литературного произведения
"""

from flask import Flask, render_template, request
from lexicon_loader import LexiconLoader
from preprocessor import Preprocessor
from emotion_analyzer import EmotionalProfileAnalyzer
from vad_analyzer import VADProfileAnalyzer
from cultural_analyzer import CulturalAnalyzer, CulturalDensityAnalyzer
from eqa_calculator import EQAMetricsCalculator
from llm_advisor import LLMAdvisor
from ml_emotion_analyzer import MLShiftDetector

app = Flask(__name__)

print("Загрузка системы")
print("=" * 60 + "\n")

lexicon = LexiconLoader()
preprocessor = Preprocessor()
emotion_analyzer = EmotionalProfileAnalyzer(lexicon)
vad_analyzer = VADProfileAnalyzer(lexicon)
cultural_analyzer = CulturalAnalyzer(lexicon.cultural_markers)
cultural_density_analyzer = CulturalDensityAnalyzer(cultural_analyzer)
eqa_calculator = EQAMetricsCalculator(vad_analyzer, emotion_analyzer, preprocessor)
llm_advisor = LLMAdvisor(ollama_model="qwen2.5:7b")
ml_shift_detector = MLShiftDetector()

print("\n Система готова к работе!")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    
    if request.method == "POST":
        original = request.form.get("original_text", "")
        translated = request.form.get("translated_text", "")
        
        if original and translated:
            result = analyze_pair(original, translated)
    
    return render_template("index.html", result=result)


def analyze_pair(original, translated):
    """Анализ пары текстов"""
    
    preprocessed = preprocessor.preprocess_pair(original, translated)
    
    emotion_orig = emotion_analyzer.analyze(original, 'english')
    emotion_trans = emotion_analyzer.analyze(translated, 'russian')
    
    vad_orig = vad_analyzer.analyze(original)
    vad_trans = vad_analyzer.analyze(translated)
    
    vad_distance = vad_analyzer.calculate_distance(vad_orig, vad_trans)
    ed_result = eqa_calculator.calculate_ed(vad_orig, vad_trans)
    pde_result = eqa_calculator.calculate_pde(emotion_orig, emotion_trans)
    
    cultural_markers = cultural_analyzer.find_markers_in_text(original)
    
    unique_markers = []
    seen = set()
    for m in cultural_markers:
        if m['marker'] not in seen:
            seen.add(m['marker'])
            unique_markers.append(m)
    
    k_result = eqa_calculator.calculate_k(
        preprocessed['original_sentences'],
        preprocessed['translated_sentences'],
        preprocessed['aligned'],
        cultural_markers
    )
    
    eqa_result = eqa_calculator.calculate_eqa(ed_result, pde_result, k_result)
    
    density_profile = cultural_density_analyzer.calculate_profile(original)
    density_plot = cultural_density_analyzer.create_plot(density_profile)
    
    marker_sentences = cultural_analyzer.find_marker_sentences(original, preprocessed['original_sentences'])
    
    # Генерация рекомендаций через LLM
    recommendation = llm_advisor.generate_recommendation(
        original[:500], translated[:500],
        eqa_result['eqa_score'],
        ed_result['normalized_distance'],
        pde_result['pde_score'],
        k_result['k_score'],
        emotion_orig['dominant_russian'],
        emotion_trans['dominant_russian']
    )
    
    # Генерация рекомендаций для проблемных маркеров
    enhanced_problematic = []
    for pm in k_result.get('problematic_markers', [])[:5]:
        marker_rec = llm_advisor.generate_marker_recommendation(
            pm['marker'],
            pm['original_sentence'],
            pm['translated_sentence'],
            pm['loss']
        )
        enhanced_problematic.append({
            'marker': pm['marker'],
            'original_sentence': pm['original_sentence'],
            'translated_sentence': pm['translated_sentence'],
            'loss': pm['loss'],
            'similarity': pm['similarity'],
            'llm_recommendation': marker_rec
        })
    
    # ML-анализ эмоционального фона
    ml_analysis = ml_shift_detector.analyze_emotional_shift_ml(original, translated)
    
    lexical_score = 1 - ed_result['normalized_distance']
    final_emotion_score = ml_shift_detector.calculate_final_emotion_score(
        lexical_score,
        ml_analysis['ml_similarity']
    )
    
    return {
        'emotion_original': emotion_orig,
        'emotion_translated': emotion_trans,
        'vad_original': vad_orig,
        'vad_translated': vad_trans,
        'vad_distance': vad_distance,
        'ed_metric': ed_result,
        'pde_metric': pde_result,
        'k_metric': {
            'score': k_result['k_score'],
            'interpretation': k_result['interpretation'],
            'problematic_count': len(k_result.get('problematic_markers', []))
        },
        'eqa_metric': eqa_result,
        'cultural': {
            'total_markers': density_profile['total_markers'],
            'density_plot': density_plot,
            'markers': [{'marker': m['marker'], 'context': m['context']} for m in unique_markers[:15]],
            'problematic_markers': enhanced_problematic
        },
        'llm_recommendation': recommendation,
        'ml_analysis': ml_analysis,
        'final_emotion_score': final_emotion_score,
        'preprocessing': {
            'original_words': len(preprocessed['original_clean'].split()),
            'translated_words': len(preprocessed['translated_clean'].split()),
            'original_sentences': len(preprocessed['original_sentences']),
            'translated_sentences': len(preprocessed['translated_sentences']),
            'aligned_pairs': len(preprocessed['aligned'])
        }
    }


if __name__ == "__main__":
    print("Приложение запущено по адресу: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)