import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("D1-NPK.csv")
# use required features
cdf = df[['pH', 'CEC', 'LIME_INDEX', 'TEMPERATURE', 'PHOSPHORUS']]


# Training Data and Predictor Variable
# Use all data for training (train-test-split not used)
x = cdf.iloc[:, :4]
y = cdf.iloc[:, -1]
regressor = LinearRegression()

# Fitting model with trainig data
regressor.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model_for_PHOSPHORUS_prediction.pkl', 'wb'))
'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''
