from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():

    artists = pd.read_csv('brief-19-01-recommenderSystem/data/artists.dat', sep='\t', usecols=['name'])

    listed = tuple(artists['name'])[:50]

    return render_template('index.html', listed = listed)

@app.route('/result', methods = ['POST'])
def result():
    r = request.form.get("selection")
    print(r)
    return render_template('resultat.html', selection = r)

if __name__ == "__main__":
    app.run()
