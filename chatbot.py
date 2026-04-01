from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

faq = {
    "hi": "Hello!",
    "how are you": "I am fine, thank you!",
    "what is your name": "I am a chatbot.",
    "what is ai": "AI means Artificial Intelligence.",
    "bye": "Goodbye!"
}

questions = list(faq.keys())
answers = list(faq.values())

def chatbot_response(user_input):
    all_text = questions + [user_input]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_text)
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    index = similarity.argmax()
    return answers[index]

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form["message"]
        response = chatbot_response(user_input.lower())
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)