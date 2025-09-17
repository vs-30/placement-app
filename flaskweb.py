import sys
import logging
import os
from flask import Flask, render_template, url_for, flash, redirect, request, jsonify
from forms import RegistrationForm, LoginForm
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from werkzeug.utils import secure_filename

# ------------------- LOGGING -------------------
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
print("✅ Logging initialized")

# ------------------- MONGO DB SETUP -------------------
mongo_uri = os.getenv("MONGO_URI")  # set this in Render Environment
try:
    client = MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=5000
    )
    client.server_info()
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

# ------------------- UPLOAD SETTINGS -------------------
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------- RESUME CHECKER -------------------
import fitz  # PyMuPDF
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

role_skills = {
    "Software Developer / Full Stack Developer": "java python c++ javascript react nodejs html css sql system design dsa",
    "Software Tester / QA Engineer": "manual testing automation selenium junit testng bug tracking quality assurance python java",
    "Database Engineer / Data Analyst": "sql mysql oracle database design normalization queries etl data analysis pandas statistics",
    "AI/ML Engineer / Data Scientist": "python machine learning deep learning ai tensorflow pytorch sql statistics data analysis numpy pandas"
}

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

def extract_text_from_pdf_safe(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            content = page.get_text("text")
            if content:
                text += content.lower() + " "
    return text

def extract_text_from_docx_safe(file_path):
    doc = docx.Document(file_path)
    return " ".join([para.text.lower() for para in doc.paragraphs])

def calculate_similarity(resume_text, reference_text):
    if not reference_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, reference_text])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

# ------------------- COMPANY QUIZ -------------------
companies = {
    "Amazon": 0, "Google": 0, "PayPal": 0, "TCS": 0,
    "Wipro": 0, "Accenture": 0, "Deloitte": 0,
    "Wells Fargo": 0, "Fidelity": 0, "Roche": 0
}

rules = {
    "work_style": {"A": ["Amazon","PayPal","Google"], "B": ["TCS","Wipro","Accenture"],
                   "C": ["Wells Fargo","Fidelity"], "D": ["Roche","Deloitte"]},
    "tech_interest": {"A": ["Amazon","Google","Accenture"], "B": ["Wells Fargo","Fidelity","PayPal"],
                      "C": ["Deloitte","Accenture"], "D": ["Roche"], "E": ["TCS","Wipro"]},
    "career_goal": {"A": ["Amazon","Google","PayPal"], "B": ["TCS","Wipro","Accenture"],
                    "C": ["Wells Fargo","Fidelity","Roche"], "D": ["Deloitte","Accenture"]},
    "skills_focus": {"A": ["Amazon","Google","PayPal"], "B": ["TCS","Wipro","Accenture"],
                     "C": ["Wells Fargo","Fidelity"], "D": ["Roche"], "E": ["Deloitte","Accenture"]},
    "culture": {"A": ["Amazon","Google"], "B": ["TCS","Wipro","Accenture","Deloitte"],
                "C": ["Wells Fargo","Fidelity","Roche"], "D": ["PayPal","Google","Amazon"]},
    "salary": {"A": ["Amazon","Google","PayPal"], "B": ["Deloitte","Accenture"],
               "C": ["Wells Fargo","Fidelity","Roche"], "D": ["TCS","Wipro"]}
}

def recommend_company(answers):
    scores = {c: 0 for c in companies.keys()}
    for q, ans in answers.items():
        if q in rules and ans in rules[q]:
            for company in rules[q][ans]:
                scores[company] += 1
    best_company = max(scores, key=scores.get)
    return best_company, scores

# ------------------- ROUTES -------------------
@app.route("/")
def home():
    return render_template('home.html', show_sidebar=True)

@app.route("/about")
def about():
    return render_template('about.html', show_sidebar=False)

@app.route("/login", methods=["GET","POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = users.find_one({"email": form.email.data})
        if user and bcrypt.check_password_hash(user["password"], form.password.data):
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        flash("Invalid email or password", "danger")
    return render_template("login.html", form=form)

@app.route("/register", methods=["GET","POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        users.insert_one({
            "username": form.username.data,
            "email": form.email.data,
            "password": hashed_password
        })
        flash("Account created! You can now log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", form=form)

@app.route("/quiz")
def quiz_home():
    return render_template("quiz_home.html", show_sidebar=True)

@app.route("/quiz/company", methods=["GET","POST"])
def company_quiz():
    if request.method == "POST":
        answers = {
            "work_style": request.form.get("work_style"),
            "tech_interest": request.form.get("tech_interest"),
            "career_goal": request.form.get("career_goal"),
            "skills_focus": request.form.get("skills_focus"),
            "culture": request.form.get("culture"),
            "salary": request.form.get("salary")
        }
        company, scores = recommend_company(answers)
        return render_template("company_result.html", company=company, scores=scores)
    return render_template("company_quiz.html", show_sidebar=True)

@app.route("/quiz/role")
def role_quiz():
    return render_template("quiz.html", show_sidebar=True)

@app.route("/resume_checker", methods=["GET","POST"])
def resume_checker():
    if request.method == "POST":
        role = request.form.get("role")
        company = request.form.get("company")
        file = request.files.get("resume")
        if not file or not allowed_file(file.filename):
            return render_template("resume_checker.html", error="Upload PDF/DOCX under 5 MB")
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        try:
            if filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf_safe(save_path)
            else:
                resume_text = extract_text_from_docx_safe(save_path)
            role_score = calculate_similarity(resume_text, role_skills.get(role,""))
            company_score = calculate_similarity(resume_text, company_skills.get(company,""))
            overall_score = calculate_similarity(resume_text, overall_text)
            os.remove(save_path)
            return render_template("resume_checker.html",
                                   role=role, company=company,
                                   role_score=role_score,
                                   company_score=company_score,
                                   overall_score=overall_score)
        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)
            return render_template("resume_checker.html", error=f"Error processing file: {str(e)}")
    return render_template("resume_checker.html")

@app.route("/save_results", methods=["POST"])
def save_results():
    data = request.json
    user_id = data.get("user_id")
    users.update_one(
        {"user_id": user_id},
        {"$set": {
            "answers": data.get("answers"),
            "preferred_role": data.get("preferred"),
            "suggested_role": data.get("suggested"),
            "final_message": data.get("finalMessage"),
            "dream_company": data.get("dreamCompany", "Unknown")
        }},
        upsert=True
    )
    return jsonify({"status":"success"})

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
