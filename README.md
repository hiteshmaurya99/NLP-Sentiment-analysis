# NLP-Sentiment-analysis
Colab notebook showcasing the implementation on NLP on binary classification task.
Commencing with the foundational steps, the Kaggle package was installed to facilitate seamless access to datasets. This was followed by the establishment of secure Kaggle API credentials, ensuring a structured and secure approach to data retrieval.

`python
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json`
Subsequently, the sentiment140 dataset was acquired from Kaggle through the command:

`python
!kaggle datasets download -d kazanova/sentiment140`
Once in possession of the dataset, a concise code snippet was utilized to extract its contents:

`python
from zipfile import ZipFile
dataset = '/content/sentiment140.zip'
with ZipFile(dataset, 'r') as zip:
    zip.extractall()`
Moving forward, attention was directed towards data preprocessing. Initial checks for missing values were conducted, and adjustments were made to target values to establish a binary sentiment classification.

`python
twitter_data.isna().sum()
twitter_data.replace({'target': {4: 1}}, inplace=True)`
The text data underwent further refinement through a process of word stemming. This involved reducing words to their root forms, thereby enhancing the model's comprehension.

`python
port_stem = PorterStemmer()`
`python
def stemming(content):
    # ... (code for stemming function)`
`python    
twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)`
Transformation of the text data into TF-IDF vectors ensued, a pivotal step in converting words into numerical representations conducive for model training.

`python
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)`
The spotlight then shifted to model training, with Logistic Regression taking center stage:

`python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)`
Model evaluation metrics were computed to gauge performance on both training and test datasets:

`python
training_data_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_data_accuracy = accuracy_score(Y_test, model.predict(X_test))`
Persisting the trained model for future use involved the application of the pickle library:

`python
import pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))`
Concluding the process, the saved model was reloaded for testing on a fresh piece of Twitter text:

`python
loaded_model = pickle.load(open(model_path, 'rb'))
prediction = loaded_model.predict(X_new)`
In summation, from establishing the data pipeline to training a sentiment analysis model, each step was meticulously executed, culminating in a powerful tool capable of discerning Twitter sentiments.

Outcome: Achieved 77.8% accuracy on test data for sentiment analysis. Model saved for future use.
