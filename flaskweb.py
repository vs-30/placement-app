import sys
import logging
import os
import pandas as pd
import requests
import threading
import csv
from io import StringIO
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
mongo_uri = os.getenv("MONGO_URI")  # set in Render Environment
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


# ------------------- CSV INGEST FUNCTION -------------------
def ingest_csv_from_url(url, collection):
    """
    Fetch a CSV file from a URL and insert its contents into the given MongoDB collection.
    Handles missing collection, cleans fields, and avoids duplicates using 'ID' or 'slug'.
    """
    if collection is None:
        logging.warning(f"❌ Collection is None. Cannot insert data from {url}")
        return

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        content = response.content.decode("utf-8")

        reader = csv.DictReader(content.splitlines())
        docs = []

        for row in reader:
            # Clean up and typecast fields if they exist
            if 'ID' in row:
                try:
                    row['ID'] = int(row['ID'])
                except ValueError:
                    row['ID'] = None
            if 'Frequency' in row:
                try:
                    row['Frequency'] = float(row['Frequency'])
                except ValueError:
                    row['Frequency'] = None

            for field in ['Acceptance', 'Difficulty', 'Title', 'Leetcode Question Link']:
                if field in row and row[field] is not None:
                    row[field] = row[field].strip()

            # Generate a slug for uniqueness if not provided
            slug = (row.get('Title') or "").lower().replace(" ", "-")
            slug = "".join(ch for ch in slug if ch.isalnum() or ch == "-")
            if not slug:
                continue

            # Skip duplicate entries
            if collection.find_one({"slug": slug}):
                logging.info(f"⏩ Skipping duplicate slug: {slug}")
                continue

            row['slug'] = slug
            docs.append(row)

        if docs:
            logging.info(f"Inserting {len(docs)} docs into {collection.name}")
            collection.insert_many(docs)
            logging.info(f"✅ Successfully inserted {len(docs)} docs from {url}")
        else:
            logging.warning(f"❌ No documents to insert from {url}")

    except Exception as e:
        logging.error(f"❌ Error processing CSV from {url}: {e}")


# ------------------- GITHUB CSV INGEST -------------------
GITHUB_API_URL = "https://api.github.com/repos/krishnadey30/LeetCode-Questions-CompanyWise/contents/"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # token from Render env
TARGET_COMPANIES = [
    "AMAZON", "WELLS FARGO", "FIDELITY", "PAYPAL", "ROCHE",
    "DELOITTE", "TCS", "ACCENTURE", "WIPRO", "GOOGLE", "MICROSOFT"
]


def get_company_csv_files(target_companies):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    response = requests.get(GITHUB_API_URL, headers=headers, timeout=10)
    response.raise_for_status()
    files = response.json()

    company_files = {company: [] for company in target_companies}
    for f in files:
        name_lower = f["name"].lower()
        for company in target_companies:
            if company.lower().replace(" ", "") in name_lower and f["name"].endswith(".csv"):
                company_files[company].append(f["download_url"])
    return company_files


def ingest_company_files():
    total_inserted = 0
    try:
        company_files = get_company_csv_files(TARGET_COMPANIES)
        total_files = sum(len(f) for f in company_files.values())
        print(f"✅ Found {total_files} files to ingest", flush=True)

        for company, files in company_files.items():
            print(f"📂 Starting ingestion for company: {company} ({len(files)} files)", flush=True)

            for url in files:
                inserted = ingest_csv_from_url(url, problems, company=company)  # ✅ pass company
                total_inserted += inserted

        print(f"🎯 Ingestion finished. Total new problems inserted: {total_inserted}", flush=True)

    except Exception as e:
        print("❌ Error during ingestion:", e, flush=True)


def ingest_csv_from_url(url, problems, company):
    import pandas as pd
    import logging

    try:
        df = pd.read_csv(url)
        df.columns = [c.strip().lower() for c in df.columns]

        inserted = 0

        for _, row in df.iterrows():
            problem = {
                "id": int(row.get("id", 0)) if pd.notna(row.get("id")) else 0,
                "title": str(row.get("title", "")).strip(),
                "acceptance": str(row.get("acceptance", "")).strip(),
                "difficulty": str(row.get("difficulty", "")).strip(),
                "frequency": str(row.get("frequency", "")).strip(),
                "link": str(row.get("leetcode question link", "")).strip(),
                "company_tags": [company]
            }

            if problem["title"] and problem["link"]:
                result = problems.update_one(
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




# ------------------- RESUME CHECKER -------------------
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
    "work_style": {"A": ["Amazon", "PayPal", "Google"], "B": ["TCS", "Wipro", "Accenture"],
                   "C": ["Wells Fargo", "Fidelity"], "D": ["Roche", "Deloitte"]},
    "tech_interest": {"A": ["Amazon", "Google", "Accenture"], "B": ["Wells Fargo", "Fidelity", "PayPal"],
                      "C": ["Deloitte", "Accenture"], "D": ["Roche"], "E": ["TCS", "Wipro"]},
    "career_goal": {"A": ["Amazon", "Google", "PayPal"], "B": ["TCS", "Wipro", "Accenture"],
                    "C": ["Wells Fargo", "Fidelity", "Roche"], "D": ["Deloitte", "Accenture"]},
    "skills_focus": {"A": ["Amazon", "Google", "PayPal"], "B": ["TCS", "Wipro", "Accenture"],
                     "C": ["Wells Fargo", "Fidelity"], "D": ["Roche"], "E": ["Deloitte", "Accenture"]},
    "culture": {"A": ["Amazon", "Google"], "B": ["TCS", "Wipro", "Accenture", "Deloitte"],
                "C": ["Wells Fargo", "Fidelity", "Roche"], "D": ["PayPal", "Google", "Amazon"]},
    "salary": {"A": ["Amazon", "Google", "PayPal"], "B": ["Deloitte", "Accenture"],
               "C": ["Wells Fargo", "Fidelity", "Roche"], "D": ["TCS", "Wipro"]}
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


@app.route("/ingest_companies_dynamic")
def ingest_companies_dynamic():
    print("🚀 Ingest route hit", flush=True)
    thread = threading.Thread(target=ingest_company_files, daemon=True)
    thread.start()
    return jsonify({
        "status": "success",
        "message": "Ingestion started in background. Check logs for progress."
    })


@app.route("/about")
def about():
    return render_template('about.html', show_sidebar=False)


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = users.find_one({"username": form.username.data})  # Use username, not email
        if user and bcrypt.check_password_hash(user["password"], form.password.data):
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        flash("Invalid username or password", "danger")
    return render_template("login.html", form=form)


@app.route("/ingest_problems")
def ingest_problems():
    sample_problems = [
        {
            "title": "Two Sum",
            "slug": "two-sum",
            "link": "https://leetcode.com/problems/two-sum/",
            "difficulty": "Easy",
            "tags": ["Array", "Hash Table"],
            "company_tags": ["Amazon", "Google"],
            "source": "github"
        },
        {
            "title": "Longest Substring Without Repeating Characters",
            "slug": "longest-substring-without-repeating-characters",
            "link": "https://leetcode.com/problems/longest-substring-without-repeating-characters/",
            "difficulty": "Medium",
            "tags": ["Hash Table", "String", "Sliding Window"],
            "company_tags": ["Microsoft"],
            "source": "github"
        }
    ]
    inserted = problems.insert_many(sample_problems)
    return jsonify({"status": "success", "inserted_ids": [str(i) for i in inserted.inserted_ids]})


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


@app.route("/quiz")
def quiz_home():
    return render_template("quiz_home.html", show_sidebar=True)


@app.route("/quiz/company", methods=["GET", "POST"])
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

        # Save results in session
        session["last_company"] = company
        session["last_scores"] = scores

        return redirect(url_for("company_results"))

    return render_template("company_quiz.html", show_sidebar=True)


@app.route("/quiz/company/results")
def company_results():
    company = session.get("last_company")
    scores = session.get("last_scores", {})
    return render_template("company_results.html", company=company, scores=scores, show_sidebar=True)


@app.route("/quiz/role")
def role_quiz():
    return render_template("quiz.html", show_sidebar=True)


@app.route("/resume_checker", methods=["GET", "POST"])
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
            role_score = calculate_similarity(resume_text, role_skills.get(role, ""))
            company_score = calculate_similarity(resume_text, company_skills.get(company, ""))
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
    user_email = data.get("email")
    if not user_email:
        return jsonify({"status": "error", "message": "No user email provided"}), 400

    users.update_one(
        {"email": user_email},
        {"$set": {
            "answers": data.get("answers"),
            "preferred_role": data.get("preferred"),
            "suggested_role": data.get("suggested"),
            "final_message": data.get("finalMessage"),
            "dream_company": data.get("dreamCompany", "Unknown")
        }},
        upsert=True
    )
    return jsonify({"status": "success"})


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
