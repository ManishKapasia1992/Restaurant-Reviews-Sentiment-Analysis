from flask import Flask, render_template, request
import pickle

# Loads the Counvectorizer pickle file
filename1 = r'C:\Users\admin\Desktop\Restaurant_Reviews_Sentmental_Anlysis_cv_transform.pkl'
cv = pickle.load(open(filename1, 'rb'))

# Loads the Nb_classifier pickle file
filename2 = r'C:\Users\admin\Desktop\Restaurant_Reviews_Sentmental_Anlysis_nbclassifier_model.pkl'
classifier = pickle.load(open(filename2, 'rb'))

app = Flask(__name__)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        final_data = cv.transform(data).toarray()
        my_prediction = classifier.predict(final_data)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)


