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
    sent_json = article_model.gen_sentiments(article)
    obj = json.loads(sent_json)
    bias_score = obj["bias_score"] #string
    direct_quotes = obj["direct_quotes"] #list of biased quotes
    conclusion = obj["conclusion"]
    print(conclusion)
    
    return sent_json
    
@app.route("/chatbot/api/chats/image", methods = ["POST"])
def eval_texts_image(): 
    image = request.files["image"]
    print(image)
    resp_json = text_model.gemini_eval_image(image)
    #obj = json.loads(text_model.gemini_eval_image(image))
    #summary = obj["response"]
    print(resp_json)
    return resp_json
    
@app.route("/chatbot/api/chats/text", methods=["POST"])
def eval_texts_text(): 
    text = request.form["text"]
    obj = json.loads(text_model.gemini_eval_image(text))
    summary = obj["response"]
    return render_template(
        "chatbot.html", 
        summary = summary
    )


if __name__ == "__main__": 
    app.run() 