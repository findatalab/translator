from flask import Flask, render_template, request, jsonify
from emotion import analyze_text, compare_emotions, CULTURAL_MARKERS, load_cultural_markers

app = Flask(__name__)

load_cultural_markers("cultural_markers_merged.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""
    translated_text = ""
    result = None
    
    if request.method == "POST":
        original_text = request.form.get("original_text", "")
        translated_text = request.form.get("translated_text", "")
        
        if original_text and translated_text:
            original_metrics = analyze_text(original_text)
            translation_metrics = analyze_text(translated_text)
            comparison = compare_emotions(original_metrics, translation_metrics)
            
            result = {
                "original": original_metrics,
                "translation": translation_metrics,
                "comparison": comparison
            }
    
    return render_template(
        "index.html",
        original_text=original_text,
        translated_text=translated_text,
        result=result
    )

@app.route("/check_markers", methods=["POST"])
def check_markers():
    text = request.json.get("text", "")
    metrics = analyze_text(text)
    return jsonify({
        "cultural_markers": metrics["cultural_markers"],
        "count": metrics["cultural_markers_count"],
        "word_count": metrics["word_count"]
    })

if __name__ == "__main__":
    print("Анализатор эмоций с поддержкой культурных маркеров")
    app.run(debug=True)