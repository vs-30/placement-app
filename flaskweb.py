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
from random import shuffle

# ------------------- LOGGING -------------------
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
print("✅ Logging initialized", flush=True)

# ------------------- ENVIRONMENT KEYS -------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------- MONGO DB SETUP -------------------
try:
    client = MongoClient(
        MONGO_URI,
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
except Exception as e:
    print("❌ MongoDB connection failed:", e, flush=True)
    users = None
    results = None
    problems = None

# ------------------- FLASK SETUP -------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
bcrypt = Bcrypt(app)

# ------------------- UPLOAD SETTINGS -------------------
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------- GITHUB CSV INGEST -------------------
GITHUB_API_URL = "https://api.github.com/repos/krishnadey30/LeetCode-Questions-CompanyWise/contents/"
TARGET_COMPANIES = [
    "amazon", "adobe", "airbnb", "apple", "cisco", "doordash", "paypal",
    "facebook", "goldman", "google", "linkedin", "meta", "microsoft", "netflix"
]

# ------------------- TOPIC KEYWORDS -------------------
TOPIC_KEYWORDS = {
    "arrays": ["array", "subarray", "matrix", "two sum", "maximum subarray"],
    "strings": ["string", "substring", "palindrome", "longest common prefix"],
    "linked lists": ["linked list", "singly linked list", "doubly linked list"],
    "stacks & queues": ["stack", "queue", "deque"],
    "hashing": ["hash", "hashmap", "hash set", "dictionary", "two sum"],
    "heaps / priority queues": ["heap", "priority queue", "kth largest", "merge k lists"],
    "trees": ["tree", "binary tree", "binary search tree", "bst"],
    "graphs": ["graph", "adjacency", "dfs", "bfs", "topological"],
    "dynamic programming": ["dp", "dynamic programming", "memo", "tabulation", "knapsack"],
    "greedy": ["greedy", "interval", "activity selection"],
    "backtracking": ["backtrack", "permutation", "combination", "subset"],
    "bit manipulation": ["bit", "xor", "and", "or", "mask", "single number"],
    "math / number theory": ["prime", "gcd", "lcm", "factorial", "fibonacci"],
}

def infer_topic(title):
    title_lower = title.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return topic
    return "misc"

# ------------------- CSV INGEST -------------------
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
                collection.update_one(
                    {"title": problem["title"]},
                    {"$set": problem},
                    upsert=True
                )
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
    "adobe": "javascript python html css react design ux photoshop creative suite",
    "airbnb": "python java javascript react nodejs aws cloud data analytics system design",
    "apple": "swift objective-c ios macos python machine learning hardware software integration",
    "cisco": "networking c python linux cloud security routers switches protocols",
    "doordash": "python java scala microservices APIs logistics backend distributed systems",
    "paypal": "java javascript python mysql mongodb apis rest microservices fintech security encryption",
    "facebook": "python c++ java javascript react ai ml data structures algorithms",
    "goldman": "python java c# finance trading sql risk analysis investment banking",
    "google": "python c++ java machine learning ai cloud systems algorithms data structures",
    "linkedin": "java python scala data engineering big data hadoop spark algorithms cloud",
    "meta": "python c++ java machine learning ai cloud systems data structures algorithms",
    "microsoft": "c# .net python azure cloud ai machine learning data structures algorithms",
    "netflix": "java python scala microservices cloud streaming recommendation systems ai"
}

overall_text = " ".join(list(role_skills.values()) + list(company_skills.values()))

def extract_text_from_pdf_safe(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text").lower() + " "
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

# ------------------- ROUTES -------------------
@app.route("/")
def home():
    return render_template("home.html", show_sidebar=True)

@app.route("/roadmap")
def roadmap():
    return render_template(
        "roadmap.html",
        show_sidebar=True,
        companies_list=TARGET_COMPANIES,
        topics_list=list(TOPIC_KEYWORDS.keys()),
        roadmap=None
    )

@app.route("/generate_roadmap", methods=["POST"])
def generate_roadmap():
    try:
        company = request.form.get("company", "").lower()
        weeks = int(request.form.get("weeks", 0))
        hours_per_week = int(request.form.get("hours_per_week", 0))
        selected_topics = request.form.getlist("topics")

        if not company or weeks <= 0 or hours_per_week <= 0:
            flash("Please fill all fields correctly.", "danger")
            return redirect(url_for("roadmap"))

        company_problems = list(problems.find({"company_tags": company, "title": {"$ne": ""}}))
        all_problems = list(problems.find({"title": {"$ne": ""}}))

        if selected_topics:
            company_problems = [p for p in company_problems if infer_topic(p.get("title", "")) in selected_topics]
            all_problems = [p for p in all_problems if infer_topic(p.get("title", "")) in selected_topics]

        if not company_problems:
            flash(f"No problems found for {company.capitalize()} with selected topics.", "warning")
        if not all_problems:
            flash("No problems found for selected topics.", "warning")
            return redirect(url_for("roadmap"))

        shuffle(company_problems)
        shuffle(all_problems)

        DIFFICULTY_TIME = {"easy": 10, "medium": 20, "hard": 40}
        minutes_per_week = hours_per_week * 60
        roadmap = {}
        problem_index = 0

        for week_num in range(1, weeks + 1):
            week_tasks = []
            used_minutes = 0

            while used_minutes < minutes_per_week and problem_index < len(company_problems):
                prob = company_problems[problem_index]
                difficulty = prob.get("difficulty", "medium").lower()
                expected_time = DIFFICULTY_TIME.get(difficulty, 20)
                topic = infer_topic(prob.get("title", ""))

                if used_minutes + expected_time <= minutes_per_week:
                    week_tasks.append({
                        "title": prob.get("title", "Unknown"),
                        "link": prob.get("link", "#"),
                        "difficulty": difficulty,
                        "expected_time_min": expected_time,
                        "topic": topic
                    })
                    used_minutes += expected_time
                    problem_index += 1
                else:
                    break

            roadmap[week_num] = week_tasks

        return render_template(
            "roadmap.html",
            show_sidebar=True,
            companies_list=TARGET_COMPANIES,
            topics_list=list(TOPIC_KEYWORDS.keys()),
            roadmap=roadmap,
            selected_company=company,
            selected_topics=selected_topics
        )

    except Exception as e:
        logging.error(f"Error generating roadmap: {e}")
        flash("Something went wrong while generating the roadmap.", "danger")
        return redirect(url_for("roadmap"))

# ------------------- MAIN -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
