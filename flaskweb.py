import sys
import logging
import os
import pandas as pd
import requests
import threading
from flask import Flask, render_template, url_for, flash, redirect, request, jsonify, session
from forms import RegistrationForm, LoginForm
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- LOGGING -------------------
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
print("✅ Logging initialized", flush=True)

# ------------------- MONGO DB SETUP -------------------
mongo_uri = os.getenv("MONGO_URI")
try:
    client = MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=5000
    )
    client.server_info()
    print("✅ MongoDB connection successful", flush=True)
    db = client["placementdb"]
    users = db["users"]
    results = db["results"]
    problems = db["problems"]
    problem_variants = db["problem_variants"]
except Exception as e:
    print("❌ MongoDB connection failed:", e, flush=True)
    users = None
    results = None
    problems = None

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

# ------------------- GITHUB CSV INGEST -------------------
GITHUB_API_URL = "https://api.github.com/repos/krishnadey30/LeetCode-Questions-CompanyWise/contents/"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TARGET_COMPANIES = [
    "amazon", "adobe", "airbnb", "apple", "cisco", "doordash", "paypal",
    "facebook", "goldman", "google", "linkedin", "meta", "microsoft", "netflix"
]

def get_company_csv_files(target_companies):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    response = requests.get(GITHUB_API_URL, headers=headers, timeout=10)
    response.raise_for_status()
    files = response.json()

    company_files = {company: [] for company in target_companies}
    for f in files:
        filename = f["name"].lower()
        for company in target_companies:
            if filename.startswith(company) and filename.endswith(".csv"):
                company_files[company].append(f["download_url"])
    return company_files

def ingest_csv_from_url(url, collection, company):
    try:
        df = pd.read_csv(url)
        df.columns = [c.strip() for c in df.columns]

        inserted = 0
        for _, row in df.iterrows():
            problem = {
                "id": int(row.get("ID", 0)) if pd.notna(row.get("ID")) else 0,
                "title": str(row.get("Title", "")).strip(),
                "acceptance": str(row.get("Acceptance", "")).strip(),
                "difficulty": str(row.get("Difficulty", "")).strip(),
                "frequency": str(row.get("Frequency", "")).strip(),
                "link": str(row.get("Leetcode Question Link", "")).strip(),
                "company_tags": [company]
            }
            if problem["title"] and problem["link"]:
                result = collection.update_one(
                    {"title": problem["title"]},
                    {"$set": problem},
                    upsert=True
                )
                if result.upserted_id:
                    inserted += 1

        logging.info(f"✅ Inserted {inserted} new problems for {company} from {url}")
        return inserted

    except Exception as e:
        logging.error(f"❌ Failed to ingest {url}: {e}")
        return 0

def ingest_company_files():
    total_inserted = 0
    try:
        company_files = get_company_csv_files(TARGET_COMPANIES)
        total_files = sum(len(f) for f in company_files.values())
        print(f"✅ Found {total_files} files to ingest", flush=True)

        for company, files in company_files.items():
            print(f"📂 Starting ingestion for company: {company} ({len(files)} files)", flush=True)
            for url in files:
                inserted = ingest_csv_from_url(url, problems, company)
                total_inserted += inserted

        print(f"🎯 Ingestion finished. Total new problems inserted: {total_inserted}", flush=True)
    except Exception as e:
        print("❌ Error during ingestion:", e, flush=True)

# ------------------- RESUME CHECKER -------------------
role_skills = {
    "Software Developer / Full Stack Developer": "java python c++ javascript react nodejs html css sql system design dsa",
    "Software Tester / QA Engineer": "manual testing automation selenium junit testng bug tracking quality assurance python java",
    "Database Engineer / Data Analyst": "sql mysql oracle database design normalization queries etl data analysis pandas statistics",
    "AI/ML Engineer / Data Scientist": "python machine learning deep learning ai tensorflow pytorch sql statistics data analysis numpy pandas"
}

company_skills = {
    "amazon": "python sql machine learning aws system design data structures algorithms",
    "wells fargo": "java python c# sql oracle azure finance cybersecurity compliance",
    "fidelity": "java python sql aws azure devops agile finance investment concepts",
    "paypal": "java javascript python mysql mongodb apis rest microservices fintech security encryption",
    "roche": "python r sql data analysis healthcare cloud ai ml bioinformatics",
    "deloitte": "java python sql javascript sap salesforce cloud data analytics cybersecurity communication",
    "tcs": "c java python aptitude dbms software testing coding",
    "accenture": "java python c sql aws azure gcp salesforce devops sap teamwork",
    "wipro": "c java python aptitude logical reasoning ai ml testing cloud"
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
companies = {c: 0 for c in company_skills.keys()}

rules = {
    "work_style": {"A": ["amazon", "paypal", "google"], "B": ["tcs", "wipro", "accenture"],
                   "C": ["wells fargo", "fidelity"], "D": ["roche", "deloitte"]},
    "tech_interest": {"A": ["amazon", "google", "accenture"], "B": ["wells fargo", "fidelity", "paypal"],
                      "C": ["deloitte", "accenture"], "D": ["roche"], "E": ["tcs", "wipro"]},
    "career_goal": {"A": ["amazon", "google", "paypal"], "B": ["tcs", "wipro", "accenture"],
                    "C": ["wells fargo", "fidelity", "roche"], "D": ["deloitte", "accenture"]},
    "skills_focus": {"A": ["amazon", "google", "paypal"], "B": ["tcs", "wipro", "accenture"],
                     "C": ["wells fargo", "fidelity"], "D": ["roche"], "E": ["deloitte", "accenture"]},
    "culture": {"A": ["amazon", "google"], "B": ["tcs", "wipro", "accenture", "deloitte"],
                "C": ["wells fargo", "fidelity", "roche"], "D": ["paypal", "google", "amazon"]},
    "salary": {"A": ["amazon", "google", "paypal"], "B": ["deloitte", "accenture"],
               "C": ["wells fargo", "fidelity", "roche"], "D": ["tcs", "wipro"]}
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
    return render_template('home.html', show_sidebar=True, TARGET_COMPANIES=TARGET_COMPANIES)

@app.route("/quiz")
def quiz_home():
    return render_template("quiz.html", show_sidebar=True)

@app.route("/ingest_companies_dynamic")
def ingest_companies_dynamic():
    thread = threading.Thread(target=ingest_company_files, daemon=True)
    thread.start()
    return jsonify({"status": "success", "message": "Ingestion started in background. Check logs for progress."})

@app.route("/about")
def about():
    return render_template('about.html', show_sidebar=False)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = users.find_one({"username": form.username.data})
        if user and bcrypt.check_password_hash(user["password"], form.password.data):
            flash("Login successful!", "success")
            session['username'] = user['username']
            return redirect(url_for("home"))
        flash("Invalid username or password", "danger")
    return render_template("login.html", form=form)

@app.route("/register", methods=["GET", "POST"])
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

@app.route("/check_resume", methods=["POST"])
def check_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf_safe(filepath)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx_safe(filepath)
    else:
        text = ""

    similarity = calculate_similarity(text, overall_text)
    return jsonify({"similarity": similarity})

# ------------------- MAIN -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
