import sys
import logging
import os
from flask import Flask, render_template, url_for, flash, redirect, session, request, jsonify
from forms import RegistrationForm, LoginForm
from flask_bcrypt import Bcrypt
from pymongo import MongoClient

# ------------------- LOGGING -------------------
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
print("✅ Logging initialized")

# ------------------- MONGO DB SETUP -------------------
mongo_uri = os.getenv("MONGO_URI")  # set this in Render Environment

try:
    client = MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,  # fix SSL handshake on Render
        serverSelectionTimeoutMS=5000       # 5 sec timeout
    )
    client.server_info()  # force connection to check
    print("✅ MongoDB connection successful")
    db = client["placementdb"]
    users = db["users"]
    results = db["results"]
except Exception as e:
    print("❌ MongoDB connection failed:", e)
    users = None
    results = None

# ------------------- FLASK SETUP -------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "dev_secret_key")
bcrypt = Bcrypt(app)

# ------------------- RESUME CHECKER ML SETUP -------------------
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Roles and required skills
role_skills = {
    "Software Developer / Full Stack Developer": "java python c++ javascript react nodejs html css sql system design dsa",
    "Software Tester / QA Engineer": "manual testing automation selenium junit testng bug tracking quality assurance python java",
    "Database Engineer / Data Analyst": "sql mysql oracle database design normalization queries etl data analysis pandas statistics",
    "AI/ML Engineer / Data Scientist": "python machine learning deep learning ai tensorflow pytorch sql statistics data analysis numpy pandas"
}

# Companies and required skills
company_skills = {
    "Amazon": "python sql machine learning aws system design data structures algorithms",
    "Wells Fargo": "java python c# sql oracle azure finance cybersecurity compliance",
    "Fidelity": "java python sql aws azure devops agile finance investment concepts",
    "Paypal": "java javascript python mysql mongodb apis rest microservices fintech security encryption",
    "Roche": "python r sql data analysis healthcare cloud ai ml bioinformatics",
    "Deloitte": "java python sql javascript sap salesforce cloud data analytics cybersecurity communication",
    "TCS": "c java python aptitude dbms software testing coding",
    "Accenture": "java python c sql aws azure gcp salesforce devops sap teamwork",
    "Wipro": "c java python aptitude logical reasoning ai ml testing cloud"
}

overall_text = " ".join(list(role_skills.values()) + list(company_skills.values()))

def extract_text_from_pdf(pdf_path):
    """Extract raw text from PDF resumes"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content.lower() + " "
    return text

def calculate_similarity(resume_text, reference_text):
    """Return similarity score between resume and reference skills"""
    if not reference_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, reference_text])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

# ------------------- ROUTES -------------------

@app.route("/")
def home():
    print("👉 Home route reached")
    return render_template('home.html', show_sidebar=True)

@app.route("/about")
def about():
    return render_template('about.html', show_sidebar=False)

@app.route("/login", methods=["GET", "POST"])
def login():
    print("👉 Login route reached")
    form = LoginForm()
    if form.validate_on_submit():
        print("👉 Form submitted")
        try:
            user = users.find_one({"email": form.email.data})
            print("👉 User query result:", user)
            if user and bcrypt.check_password_hash(user["password"], form.password.data):
                print("👉 Password matched")
                flash("Login successful!", "success")
                return redirect(url_for("home"))
            else:
                print("👉 Invalid credentials")
                flash("Invalid email or password", "danger")
                return redirect(url_for("login"))
        except Exception as e:
            print("❌ Login error:", e)
            flash("An error occurred during login", "danger")
            return redirect(url_for("login"))

    return render_template("login.html", form=form)

@app.route("/register", methods=["GET", "POST"])
def register():
    print("👉 Register route reached")
    form = RegistrationForm()
    if form.validate_on_submit():
        print("👉 Form submitted")
        try:
            hashed_password = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
            print("👉 Password hashed successfully")

            users.insert_one({
                "username": form.username.data,
                "email": form.email.data,
                "password": hashed_password
            })
            print("👉 User inserted into MongoDB")

            flash("Your account has been created! You can now log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            print("❌ Registration error:", e)
            flash("An error occurred during registration", "danger")
            return redirect(url_for("register"))

    return render_template("register.html", form=form)

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

@app.route("/resume_checker", methods=["GET", "POST"])
def resume_checker():
    if request.method == "POST":
        role = request.form.get("role")
        company = request.form.get("company")
        file = request.files.get("resume")

        if not file:
            return render_template("resume_checker.html", error="Please upload a resume")

        file_path = "temp_resume.pdf"
        file.save(file_path)
        resume_text = extract_text_from_pdf(file_path)
        os.remove(file_path)

        role_score = calculate_similarity(resume_text, role_skills.get(role, ""))
        company_score = calculate_similarity(resume_text, company_skills.get(company, ""))
        overall_score = calculate_similarity(resume_text, overall_text)

        return render_template(
            "resume_checker.html",
            role=role,
            company=company,
            role_score=role_score,
            company_score=company_score,
            overall_score=overall_score,
        )

    # GET request → show form
    return render_template("resume_checker.html")

@app.route("/save_results", methods=["POST"])
def save_results():
    data = request.json
    user_id = data.get("user_id")
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

# ------------------- MAIN -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

