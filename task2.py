import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    'text': [
        'I love this product',
        'This is amazing',
        'I hate this',
        'This is bad',
        'Very good',
        'Very poor'
    ],
    'label': ['Positive', 'Positive', 'Negative', 'Negative', 'Positive', 'Negative']
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Train model
model = MultinomialNB()
model.fit(X, df['label'])

# Test
test = ["I love it"]
test_vec = vectorizer.transform(test)

prediction = model.predict(test_vec)

print("Prediction:", prediction[0])