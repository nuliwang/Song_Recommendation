import pandas as pd
from flask import Flask, request, render_template
from sklearn.neighbors import NearestNeighbors


app = Flask(__name__)


df = pd.read_csv('song_dataset.csv')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)


user_song_matrix = df.pivot_table(index='user', columns='song', values='play_count').fillna(0)


model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(user_song_matrix)

all_songs = df['title'].unique().tolist()


def recommend_songs(user_id, data, model, original_df, listened_song, n_recommendations=5):
    if user_id not in data.index:
        return ["User ID not found."]
    if listened_song not in original_df['title'].values:
        return ["Song not found."]
    
    user_index = data.index.tolist().index(user_id)
    distances, indices = model.kneighbors(data.iloc[user_index, :].values.reshape(1, -1), n_neighbors=n_recommendations+10)
    
    recommendations = []
    for i in range(1, len(distances.flatten())):
        song_id = data.columns[indices.flatten()[i]]
        if len(recommendations) < n_recommendations:
            song_info = original_df[original_df['song'] == song_id].iloc[0]
            if song_info['title'] != listened_song:
                recommendations.append({
                    'title': song_info['title'],
                    'release': song_info['release'],
                    'artist_name': song_info['artist_name'],
                    'year': song_info['year']
                })
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        user_id = request.form['user_id']
        listened_song = request.form['listened_songs']
        recommendations = recommend_songs(user_id, user_song_matrix, model_knn, df, listened_song)
    return render_template('index.html', all_songs=all_songs, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
