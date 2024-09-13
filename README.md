Instagram Sentiment Analysis and Rating Prediction
This project demonstrates sentiment analysis and rating prediction using a dataset of Instagram reviews. The analysis involves text preprocessing, sentiment categorization, and building a neural network model to predict ratings based on review text.

Project Overview
The project involves the following steps:

Sentiment Analysis: Using NLP techniques to categorize review sentiments as Positive, Negative, or Neutral.
Text Preprocessing: Cleaning and preparing the text data for analysis.
Rating Prediction: Developing a neural network model to predict the rating of reviews based on their content.
Prerequisites
To run the project, ensure you have the following packages installed:

Python 3.6 or higher
NLTK (Natural Language Toolkit)
Pandas
NumPy
TensorFlow
scikit-learn
Matplotlib
Install the required packages using the following command:

bash
Copy code
pip install nltk pandas numpy tensorflow scikit-learn matplotlib
Data Preprocessing
The dataset consists of Instagram reviews, which are loaded from a CSV file. The following preprocessing steps are performed:

Text Cleaning: Removal of mentions, hashtags, hyperlinks, punctuation, and emojis.
Stop Words Removal: Common stop words are removed from the text.
Sentiment Analysis: The sentiment of each review is analyzed and categorized as Positive, Negative, or Neutral.

def cleantext(text):
    # Text cleaning function here
    return cleaned_text
Sentiment Analysis
The sentiment of the cleaned review text is analyzed using the TextBlob library. The sentiment is categorized into Positive, Negative, or Neutral based on the polarity score.

Rating Prediction Model
A neural network model is built using TensorFlow and Keras to predict the rating based on the sentiment and review description.

TF-IDF Vectorization: The review text is converted into TF-IDF features.
Neural Network: A multi-layer neural network is trained to predict ratings.
Early Stopping: The training process is optimized using early stopping to prevent overfitting.
python
Copy code
model = keras.Sequential([
    # Model architecture here
])

model.compile(optimizer='adam', loss='mean_squared_error')
Model Evaluation
The trained model is evaluated on a test set using various metrics:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R-squared (R²)
The loss during training and validation is plotted over epochs to visualize the model's performance.


# Model evaluation and plotting
Results
The final model achieves the following performance on the test set:

MSE: 1.22
RMSE: 1.11
MAE: 0.91
R²: 0.42
Visualization
A line chart is provided to visualize the training and validation loss over the epochs, allowing for easy assessment of the model's convergence.


License
This project is licensed under the MIT License 
