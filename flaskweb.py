from flask import Flask, render_template, url_for, flash, redirect, session,request,jsonify
from forms import RegistrationForm, LoginForm
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from urllib.parse import quote_plus

app = Flask(__name__)
app.config['SECRET_KEY'] = '6bf7cf12d543b52afc886ddb'

# --- MongoDB Atlas Setup ---
from flask import Flask, render_template, url_for, flash, redirect, session, request, jsonify
from forms import RegistrationForm, LoginForm
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from urllib.parse import quote_plus
import os  # <-- add this import for environment variables

app = Flask(__name__)
app.config['SECRET_KEY'] = '6bf7cf12d543b52afc886ddb'

# --- MongoDB Atlas Setup using environment variables ---
username = os.environ.get("MONGO_USER")
password = os.environ.get("MONGO_PASS")

encoded_username = quote_plus(username)
encoded_password = quote_plus(password)

client = MongoClient(
    f"mongodb+srv://{encoded_username}:{encoded_password}@cluster0.wzexsa3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
db = client["quiz_app"]
users = db["users"]
results = db["results"]

# --- Bcrypt for password hashing ---
bcrypt = Bcrypt(app)

# --- Bcrypt for password hashing ---
bcrypt = Bcrypt(app)

# ------------------- ROUTES -------------------

@app.route("/save_results", methods=["POST"])
def save_results():
    data = request.json
    user_id = data.get("user_id")  # make sure your frontend sends user_id
    dream_company = data.get("dreamCompany", "Unknown")

    users.update_one(
        {"user_id": user_id},
        {"$set": {
            "answers": data.get("answers"),
            "preferred_role": data.get("preferred"),
            "suggested_role": data.get("suggested"),
            "final_message": data.get("finalMessage"),
            "dream_company": dream_company
        }},
        upsert=True
    )

    return jsonify({"status": "success"})


@app.route("/")
def home():
    return render_template('home.html', show_sidebar=True)

@app.route("/about")
def about():
    return render_template('about.html', show_sidebar=False)

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        # find user in DB
        user = users.find_one({"username": username})

        if user and bcrypt.check_password_hash(user["password"], password):
            session["username"] = username
            flash(f"Welcome back {username}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "danger")

    return render_template('login.html', form=form)

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data

        # check if user already exists
        if users.find_one({"username": username}):
            flash("Username already taken. Please choose another.", "danger")
        else:
            hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
            users.insert_one({
                "username": username,
                "email": email,
                "password": hashed_pw
            })
            flash(f"Account successfully created for {username}!", "success")
            return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route("/quiz")
def quiz():
    return render_template("quiz.html", show_sidebar=True)

@app.route("/roadmap")
def roadmap():
    return render_template("roadmap.html", show_sidebar=True)

@app.route("/self_confidence")
def self_confidence():
    return render_template("self_confidence.html", show_sidebar=True)

@app.route("/company_specific")
def company_specific():
    return render_template("company_specific.html", show_sidebar=True)

# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
