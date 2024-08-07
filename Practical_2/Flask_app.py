# name:variance  # Name Of the Parameters
# in: query      # Replace with some value
# type: number   # type of variable
# required: true # If this is compulsory or not
# description: The Output Values # Suppose I get 200 , we will have to display the particular description



from flask import Flask,request #request(Get and Post Purpose)
import pandas as pd
import numpy as np
import pandas as pd 
import pandas
import pickle
import flasgger
from flasgger import Swagger # It is an API thing(flasgger)

app=Flask(__name__)
Swagger(app) # It is giving an indication to flask to generate the Ui Part



pickle_in=open('classifier.pkl','rb') # Load Classifier.pickle file in read byte mode
classifier=pickle.load(pickle_in)  # Load the file

# If I have to work this inside the flask, we will use a decorator
@app.route("/")  # This will be my route path ("/")
def Welcome():
    return "Welcome all"

# We will provide  parameters

@app.route('/predict',methods=["Get"]) # UI DEVELOPMENT
def predict_note_authentication(): # Default Template
        """Let's Authenticate the Bank Note 
        This is using docstrings for specification    
        ---
        parameters:
            - name: variance  
              in: query      
              type: number   
              required: true 
            - name: skewness
              in: query
              type: number
              required: true
            - name: curtosis
              in: query
              type: number
              required: true
            - name: entropy
              in: query
              type: number
              required: true  
        responses:
            200:
                description: The Output Values 
                
        """
        variance=request.args.get("variance") 
        skewness=request.args.get("skewness")
        curtosis=request.args.get("curtosis")
        entropy=request.args.get("entropy")
        # Prediction
        Prediction=classifier.predict([[variance,skewness,curtosis,entropy]]) # Passing the Inputs
        print(Prediction)
        return "The predicted Value is " + str(Prediction)


# I will passing a test file
@app.route('/predict_test_file',methods=["POST"])  # Run This in Postman
def predict_file():
        """Let's Authenticate the Bank's Note 
        This is using docstrings for specifications.
        ---
        parameters:
            - name: file
              in: formData
              type: file
              required: true
        responses:
            200:
                description : The Output Values

        """
        df_test=pd.read_csv(request.files.get("file"))
        print(df_test.head())
        Prediction=classifier.predict(df_test)
        return  "The predicted Value is " + str(list(Prediction))
        


if __name__=="__main__":  # Flask app will start execution from here
    app.run(debug=True,host="0.0.0.0",port=5000)

    # http://127.0.0.1:5000/apidocs/