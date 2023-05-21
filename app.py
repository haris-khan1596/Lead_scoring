from flask import Flask,request,jsonify,render_template
import joblib
import numpy as np

# Create Flask App
app = Flask(__name__,template_folder='templates')

# Load pickle model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
# Retrieve the form data# Get the feature values from the form
    lead_origin = int(request.form['Lead_Origin'])
    lead_source = int(request.form['Lead_Source'])
    do_not_email = int(request.form['Do_Not_Email'])
    do_not_call = int(request.form['Do_Not_Call'])
    total_visits = float(request.form['total_visits'])
    time_spent = float(request.form['time_spent'])
    page_views = float(request.form['page_views'])
    last_activity = int(request.form['Last_Activity'])
    country = int(request.form['Country'])
    specialization = int(request.form['Specialization'])
    current_occupation = int(request.form['What_is_your_current_occupation'])
    course_preference = int(request.form['What_matters_most_to_you_in_choosing_a_course'])
    search = int(request.form['Search'])
    newspaper_article = int(request.form['Newspaper_Article'])
    education_forums = int(request.form['X_Education_Forums'])
    newspaper = int(request.form['Newspaper'])
    digital_advertisement = int(request.form['Digital_Advertisement'])
    through_recommendations = int(request.form['Through_Recommendations'])
    city = int(request.form['City'])
    mastering_interview = int(request.form['A_free_copy_of_Mastering_The_Interview'])

    # Create a feature array
    features = np.array([lead_origin, lead_source, do_not_email,do_not_call, 
                        total_visits,time_spent,page_views,
                        last_activity, country,
                        specialization, current_occupation, course_preference, search,
                        newspaper_article, education_forums, newspaper, digital_advertisement,
                        through_recommendations, city, mastering_interview])

    # Reshape the feature array to match the model's input shape
    features = features.reshape(1, -1)

    # Make the prediction using the model
    prediction = model.predict(features)
    if prediction[0]==0:
        prediction= "Not Converted"
    else:
        prediction= "Converted"

    # Return the predicted outcome
    return render_template("index.html",prediction=prediction)
    return f"The predicted outcome is: {prediction}"
if __name__ == '__main__':
    app.run()