from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True


@app.route("/")
def dashboard():
    return render_template('dashboard.html')

@app.route("/detail")
def detail():
    return render_template("detail.html", curYear=None, curStat=None)

@app.route("/detail/<year>/<status>")
def detail_with_params(year, status):
    return render_template("detail.html", curYear=year, curStat=status)

@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route("/comparison")
def comparison():
    return render_template('comparison.html')

if __name__ == "__main__":
    app.run(debug=True)