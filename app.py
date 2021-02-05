from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import string, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel, linear_kernel

# Load in the data:-
data = pd.read_csv("datasets/clean_dataset.csv")


def clean(text):
    
    # Remove all punctuation:
    for char in text:
        if char in string.punctuation+u'\N{DEGREE SIGN}'+'039':
            text = text.replace(char,"")
    
    # Convert to lowercase:
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    text = re.sub(u'\N{DEGREE SIGN}','',text)
    text = text.lower()
    
    return text


vectorizer = TfidfVectorizer(stop_words='english',min_df=2,
	ngram_range=(1,3),strip_accents='unicode',token_pattern=r"\w+")

vec = vectorizer.fit_transform(data['genre'])
sigmoid_sim_matrix = sigmoid_kernel(vec,vec)

# Getting the indices for the animes:
indices = pd.Series(data.index, index=data['name']).drop_duplicates()

# The recommendation function:
def recommend(name, sim=sigmoid_sim_matrix):
    
    try:
	    name = clean(name)
	    
	    # Get the index corresponding to original_title
	    index = indices[name]

	    # Get the pairwsie similarity scores 
	    sim_scores = list(enumerate(sim[index]))

	    # Sort the animes accoring to their similarity scores
	    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

	    # Scores of the 10 most similar anime
	    sim_scores = sim_scores[1:11]

	    # list out the anime indices
	    anime_indices = [i[0] for i in sim_scores]

	    # Top 10 most similar movies
	    return pd.DataFrame({'Anime Name': data['name'].iloc[anime_indices].apply(lambda x: x.upper()).values,
	    							'Type':data['type'].iloc[anime_indices].values,
	                                 'Episodes':data['episodes'].iloc[anime_indices].values,
	                                 'Rating': data['rating'].iloc[anime_indices].values},
	                                 index=range(1,11))

    except:
        return False


app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
	return render_template('index.html')


@app.route('/recommendations',methods=['POST'])
def result():
	if request.method == 'POST':
		anime = str(request.form['search'])

		results = recommend(anime)

		if results is False:
			return render_template("fail.html")

		else:
			return render_template("recommendations.html",
				tables=[results.to_html(classes='data',header=True)],
				 titles=results.columns.values)

if __name__ == '__main__':
	app.run(debug=True)
