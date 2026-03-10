from flask import Flask, render_template, request
from emotion import analyze_text, compare_emotions

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)
