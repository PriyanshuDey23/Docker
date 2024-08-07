
from flask import Flask,request #request(Get and Post Purpose)
import pandas as pd
import numpy as np
import pickle


app=Flask(__name__)
pickle_in=open('Practical 1\classifier.pkl','rb') # Load Classifier.pickle file in read byte mode
classifier=pickle.load(pickle_in)  # Load the file

# If I have to work this inside the flask, we will use a decorator
@app.route("/")  # This will be my route path ("/")
def Welcome():
    return "Welcome all"

@app.route('/predict') # For running this part only(can be run in Postman) # http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=2&entropy=1 # give the parameters value
def predict_note_authentication(): # We will provide  parameters i.e.  features,
    # get the features to fit in the model
    variance=request.args.get("variance") 
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    # Prediction
    Prediction=classifier.predict([[variance,skewness,curtosis,entropy]]) # Passing the Inputs

    return "The predicted Value is " + str(Prediction)  # Concatenate the 2 values

# I will passing a test file and all the value will get predicted

# For Running Locally

@app.route('/predict_test_file')#,methods=["POST"])  
def predict_file():
    df_test=pd.read_csv("Practical 1/test.csv")
    Prediction=classifier.predict(df_test)
    return "The predicted Value is for the CSV is " + str(list(Prediction))


# For Running in Post man
"""
@app.route('/predict_test_file',methods=["POST"])  # Run This in Postman(instead of get use post)
def predict_file():
    df_test=pd.read_csv(request.files.get("file"))
    Prediction=classifier.predict(df_test)
    return "The predicted Value is for the CSV is " + str(list(Prediction))
"""









if __name__=="__main__":  # Flask app will start execution from here
    app.run()