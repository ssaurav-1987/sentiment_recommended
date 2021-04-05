import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
logreg = pickle.load(open('logreg.pkl', 'rb'))
recommender = pickle.load(open('recommender.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = int_features[0]
    if final_features in recommender.index:
        d = recommender.loc[final_features].sort_values(ascending=False)[0:20] 
        out_df = d.to_frame()
        output = fineTune(out_df.index)
        output.index = np.arange(1, len(output) + 1)
        return render_template('index.html', prediction_text='Top 5 Recommended products are :',
        tables=[output.to_html(classes='data',header=False)], titles=output.columns.values)
    else:
        return render_template('index.html', prediction_text='Username not found ')


def fineTune(product_list,top=5):
    sample_data = df[df['name'].isin(product_list)][['review','name']].copy()
    #Using the vectorizer to transform the review data 
    sample_data_vect = vectorizer.transform(sample_data['review'])
    #Predicting sentiment using the final LogReg model
    sample_data['pred_sentiment'] = logreg.predict(sample_data_vect)
    #Sorting based on the percentage positive sentiment
    final_out = sample_data.groupby(by='name').mean().sort_values('pred_sentiment',ascending=False)[:top]
    return pd.DataFrame(final_out.index).rename(columns={'name':'Product Name'})

if __name__ == "__main__":
    app.run(debug=True)