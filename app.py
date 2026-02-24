from flask import Flask, render_template, request
from trans import translator

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    translated_text = ""
    original_text = ""
    src_lang = "auto"
    dest_lang = "en"

    if request.method == "POST":
        original_text = request.form.get("text")
        src_lang = request.form.get("src_lang")
        dest_lang = request.form.get("dest_lang")

        if original_text:
            try:
                translation = translator()
                translated_text = translation
            except Exception as e:
                translated_text = f"Ошибка перевода: {str(e)}"

    return render_template(
        "index.html",
        translated_text=translated_text,
        original_text=original_text,
        src_lang=src_lang,
        dest_lang=dest_lang
    )

if __name__ == "__main__":
    app.run(debug=True)
