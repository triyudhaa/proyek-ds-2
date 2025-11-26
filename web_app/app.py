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
    # print(year, status)
    return render_template("detail.html", curYear=year, curStat=status)

@app.route("/predict/<satelit>", methods=['GET', 'POST'])
def predict(satelit):
    # ambil tanggal dari form
    if request.method == 'POST':
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        print(start_date)
        print(end_date)

        # cek tanggalnya diisi atau tidak
        if start_date == "" or end_date == "":
            error = "Tanggal mulai dan tanggal selesai harus diisi!"
            print(error)

            return render_template(
                "predict.html",
                satelit=satelit,
                error=error,
                start_date=start_date,
                end_date=end_date,
                show_segment=False
            )
        
        # jalankan model machine learning sesuai tipe satelit yang dipilih
        if satelit == 'sentinel':
            print("do sentinel model")
        else:
            print("do landsat model")

        return render_template(
            "predict.html",
            satelit=satelit,
            start_date = start_date,
            end_date = end_date,
            show_img_container = "show"
        )
    
    return render_template('predict.html', satelit=satelit)

@app.route("/comparison")
def comparison():
    return render_template('comparison.html')

if __name__ == "__main__":
    app.run(debug=True)