from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
import numpy as np

app = Flask(__name__)

artists = pd.read_csv('brief-19-01-recommenderSystem/data/artists.dat', sep='\t', usecols=['id', 'name'])
plays = pd.read_csv('brief-19-01-recommenderSystem/data/user_artists.dat', sep='\t')
ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
ap = ap.rename(columns={"weight": "playCount"}) 

artist_rank = ap.groupby(['name']) \
    .agg({'userID' : 'count', 'playCount' : 'sum'}) \
    .rename(columns={"userID" : 'totalUsers', "playCount" : "totalPlays"}) \
    .sort_values(['totalPlays'], ascending=False)

artist_rank['avgPlays'] = artist_rank['totalPlays'] / artist_rank['totalUsers']

ap = ap.join(artist_rank, on="name", how="inner").sort_values(['playCount'], ascending=False)
pc = ap.playCount
play_count_scaled = (pc - pc.min()) / (pc.max() - pc.min())
ap = ap.assign(playCountScaled=play_count_scaled)

artists_name = ap.sort_values("artistID")["name"].unique()

ratings_df = ap.pivot(index='userID', columns='artistID', values='playCountScaled')

@app.route('/')
def index():

    listed = ap['name'].unique()

    return render_template('index.html', listed = listed)

@app.route('/result', methods = ['POST'])
def result():
    if request.method == "POST":
         r = request.form.getlist("selection") 

    user_id = ratings_df.index.values
    artists_name = ap.sort_values("artistID")["name"].unique()

    new_user = max(user_id)+1
    nuser_artiste = np.zeros(len(artists_name))
    
    for i, artist in enumerate(r):
        nuser_artiste[i] = ap.playCountScaled[ap["name"]==artist].mean()

    ratings_df.loc[new_user] = nuser_artiste


    user_id = ratings_df.index.values
    ratings = ratings_df.fillna(0).values
    X = csr_matrix(ratings)
    Xcoo = X.tocoo()

    model = LightFM(learning_rate=0.08, learning_schedule='adadelta', loss='warp', random_state=42)
    model.fit(X, epochs=10, num_threads=2)

    artists_name = ap.sort_values("artistID")["name"].unique()
    n_users, n_items = ratings_df.shape

    liste_idx = list(user_id)
    idx = liste_idx.index(new_user)
    scores = model.predict(idx, np.arange(n_items))

    top_items = artists_name[np.argsort(-scores)][:10]

    reco = []
    for i in top_items:
        if i not in r:
            reco.append(i)

    return render_template('resultat.html', selection = reco)

if __name__ == "__main__":
    app.run()
