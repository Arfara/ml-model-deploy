import pickle

def load_model():
    with open('models/model_contentbased.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def get_recommendations(title, model):
    tfidf = model['tfidf']
    cosine_sim = model['cosine_sim']
    indices = model['indices']

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar food
    sim_scores = sim_scores[1:6]

    food_indices = [i[0] for i in sim_scores]
    return food_indices
