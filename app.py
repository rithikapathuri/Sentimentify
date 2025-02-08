from flask import Flask, request, render_template
app = Flask("__name__")

@app.get("/")
def index(): 
    return render_template(
        "index.html",
        title = "Welcome to my app"
        )


if __name__ == "__main__": 
    app.run() 