import json
import article_model
import text_model

from flask import Flask, request, render_template
app = Flask(__name__, static_folder='static')


@app.route("/")
def index(): 
    return render_template(
        "index.html",
        title = "Welcome to my app"
        )

@app.route("/chatbot")
def chatbot(): 
    return render_template(
        "chatbot.html",
        #title = "Welcome to my app"
        )
    
@app.route("/chatbot/api/article", methods = ["POST"])
def eval_article(): 
    print("accessed endpoint")
    article = request.form["article"]
    print("received article! ", article[0:50])
    obj = json.loads(article_model.gen_sentiments(article))
    bias_score = obj["bias_score"] #string
    direct_quotes = obj["direct_quotes"] #list of biased quotes
    conclusion = obj["conclusion"]
    
    return render_template(
        "chatbot.html", 
        bias_score = bias_score,
        direct_quotes = direct_quotes, 
        conclusion = conclusion
    )
    
@app.route("/chatbot/api/chats/image", methods = ["POST"])
def eval_texts_image(): 
    image = request.files["image"]
    print(image)
    obj = json.loads(text_model.gemini_eval_image(image))
    summary = obj["response"]
    return render_template(
        "chatbot.html", 
        summary = summary
    )
    
@app.route("/chatbot/api/chats/image", methods=["POST"])
def eval_texts_text(): 
    text = request.text 
    obj = json.loads(text_model.gemini_eval_image(text))
    summary = obj["response"]
    return render_template(
        "chatbot.html", 
        summary = summary
    )


if __name__ == "__main__": 
    app.run() 