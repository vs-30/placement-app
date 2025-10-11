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
    problem_variants = db["problem_variants"]
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
TOPICS_LIST = [
    "arrays", "strings", "linked list", "stack", "queue", "recursion",
    "dynamic programming", "greedy", "graphs", "trees", "heaps",
    "backtracking", "binary search", "math", "bit manipulation"
]
TOPIC_KEYWORDS = {
    "arrays": [
        "array", "subarray", "matrix", "two sum", "maximum subarray",
        "prefix sum", "sliding window", "rotate array", "merge intervals",
        "sort colors", "product of array except self"
    ],
    "strings": [
        "string", "substring", "palindrome", "longest common prefix",
        "anagram", "regex", "string matching", "encode", "decode",
        "compress", "valid parentheses", "reorder"
    ],
    "linked list": [
        "linked list", "singly linked list", "doubly linked list",
        "reverse linked list", "merge k lists", "cycle", "palindrome list"
    ],
    "stacks & queues": [
        "stack", "queue", "deque", "min stack", "valid parentheses",
        "sliding window maximum", "largest rectangle", "circular queue"
    ],
    "hashing": [
        "hash", "hashmap", "hash set", "dictionary", "two sum",
        "group anagrams", "contains duplicate", "subarray sum"
    ],
    "heaps / priority queues": [
        "heap", "priority queue", "kth largest", "merge k lists",
        "top k frequent", "sliding window median"
    ],
    "trees": [
        "tree", "binary tree", "binary search tree", "bst", "preorder",
        "inorder", "postorder", "level order", "height", "diameter",
        "lowest common ancestor", "symmetric tree", "path sum"
    ],
    "graphs": [
        "graph", "adjacency", "dfs", "bfs", "topological", "dijkstra",
        "floyd", "kruskal", "prim", "connected component", "cycle",
        "bipartite", "shortest path"
    ],
    "dynamic programming": [
        "dp", "dynamic programming", "memo", "tabulation", "knapsack",
        "climbing stairs", "house robber", "coin change", "longest increasing subsequence",
        "edit distance", "word break", "partition", "matrix chain"
    ],
    "greedy": [
        "greedy", "interval", "activity selection", "minimum spanning",
        "fractional knapsack", "huffman", "rearrange", "jump game"
    ],
    "backtracking": [
        "backtrack", "permutation", "combination", "subset", "n-queens",
        "sudoku", "word search", "generate parentheses", "letter case"
    ],
    "bit manipulation": [
        "bit", "xor", "and", "or", "mask", "single number",
        "count bits", "power of two", "subsets"
    ],
    "math / number theory": [
        "prime", "gcd", "lcm", "factorial", "fibonacci", "mod",
        "combinatorics", "pascal", "permutation", "combination"
    ],
    "sliding window": [
        "sliding window", "max", "min", "sum", "substring", "subarray",
        "longest", "window"
    ],
    "two pointers": [
        "two pointers", "left", "right", "pair sum", "triplet", "sorted array",
        "reverse", "partition", "container", "intersection"
    ],
    "intervals": [
        "interval", "merge", "overlap", "insert interval", "meeting rooms",
        "non-overlapping", "sort intervals"
    ],
    "design / system design": [
        "design", "LRU", "cache", "queue", "stack", "database",
        "serializer", "iterator", "heap", "deque"
    ]

}

def infer_topic(title):
    """
    Infer topic from problem title using TOPIC_KEYWORDS mapping.
    Returns 'misc' if no keyword matches.
    """
    title_lower = title.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return topic
    return "misc"

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

# Initialize companies with 0 scores based on company_skills keys
companies = {c: 0 for c in company_skills.keys()}


rules = {
    "work_style": {
        "A": ["amazon", "google", "meta"],
        "B": ["paypal", "apple", "linkedin"],
        "C": ["microsoft", "netflix"],
        "D": ["airbnb", "adobe", "doordash"],
    },
    "tech_interest": {
        "A": ["amazon", "google", "meta"],
        "B": ["paypal", "apple", "linkedin"],
        "C": ["microsoft", "netflix"],
        "D": ["airbnb", "adobe", "doordash"],
        "E": []  # if you want E option to exist but no companies, keep empty
    },
    "career_goal": {
        "A": ["amazon", "google", "paypal"],
        "B": ["meta", "microsoft", "apple"],
        "C": ["linkedin", "netflix"],
        "D": ["airbnb", "adobe", "doordash"],
    },
    "skills_focus": {
        "A": ["amazon", "google", "paypal"],
        "B": ["meta", "microsoft", "apple"],
        "C": ["linkedin", "netflix"],
        "D": ["airbnb", "adobe", "doordash"],
        "E": []
    },
    "culture": {
        "A": ["amazon", "google", "meta"],
        "B": ["paypal", "apple", "linkedin"],
        "C": ["microsoft", "netflix"],
        "D": ["airbnb", "adobe", "doordash"],
    },
    "salary": {
        "A": ["amazon", "google", "paypal"],
        "B": ["meta", "microsoft", "apple"],
        "C": ["linkedin", "netflix"],
        "D": ["airbnb", "adobe", "doordash"],
    }
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
    return render_template("home.html", show_sidebar=True)

@app.route("/quiz_home")
def quiz_home():
    return render_template("quiz_home.html", show_sidebar=True)

@app.route("/quiz")
def quiz():
    return render_template("quiz.html", show_sidebar=True)

@app.route("/ingest_companies_dynamic")
def ingest_companies_dynamic():
    print("🚀 Ingest route hit", flush=True)
    thread = threading.Thread(target=ingest_company_files, daemon=True)
    thread.start()
    return jsonify({
        "status": "success",
        "message":         "Ingestion started in background. Check logs for progress."
    })

@app.route("/about")
def about():
    return render_template("about.html", show_sidebar=False)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = users.find_one({"username": form.username.data})
        if user and bcrypt.check_password_hash(user["password"], form.password.data):
            flash("Login successful!", "success")
            session["username"] = user["username"]
            session["email"] = user["email"]  # Store email for completed questions

            # Initialize completed_questions if not present
            if "completed_questions" not in user:
                users.update_one(
                    {"email": user["email"]},
                    {"$set": {"completed_questions": {}}}
                )

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

@app.route("/update_self_confidence", methods=["POST"])
def update_self_confidence():
    """
    Called via AJAX when user clicks Submit on a question timer.
    Updates that question's completion info in MongoDB
    and recalculates self-confidence stats per topic.
    """
    try:
        data = request.get_json()
        user_email = session.get("email")
        if not user_email:
            return jsonify({"error": "User not logged in"}), 401

        question_title = data.get("question")
        topic = data.get("topic")
        difficulty = data.get("difficulty")
        expected_time = int(data.get("expected_time"))
        actual_time = int(data.get("actual_time"))

        # 1️⃣ Save completion info for that single question
        users.update_one(
            {"email": user_email},
            {"$set": {f"completed_questions.{question_title}": actual_time}},
            upsert=True
        )

        # 2️⃣ Fetch all user's completed questions
        user_data = users.find_one({"email": user_email})
        completed = user_data.get("completed_questions", {})

        # 3️⃣ Calculate topic stats
        all_problems = list(problems.find({"title": {"$ne": ""}}))
        topic_problems = [
            p for p in all_problems if infer_topic(p.get("title", "")) == topic
        ]
        total_topic = len(topic_problems)
        solved_in_time = 0
        not_in_time = []

        for prob in topic_problems:
            title = prob["title"]
            expected = DIFFICULTY_TIME.get(prob.get("difficulty", "medium").lower(), 20)
            actual = completed.get(title)
            if actual:
                if actual <= expected:
                    solved_in_time += 1
                else:
                    not_in_time.append({
                        "title": title,
                        "time": actual
                    })

        return jsonify({
            "status": "ok",
            "topic": topic,
            "solved_in_time": solved_in_time,
            "total": total_topic,
            "not_in_time": not_in_time
        })

    except Exception as e:
        print("Error in update_self_confidence:", e)
        return jsonify({"error": "Server error"}), 500


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
        session["last_company"] = company
        session["last_scores"] = scores
        return redirect(url_for("company_results"))
    return render_template("company_quiz.html", show_sidebar=True)

@app.route("/quiz/company/results")
def company_results():
    company = session.get("last_company")
    scores = session.get("last_scores", {})
    return render_template(
        "company_results.html",
        company=company,
        scores=scores,
        show_sidebar=True
    )

@app.route("/quiz/role")
def role_quiz():
    return render_template("quiz.html", show_sidebar=True)

@app.route("/submit_time", methods=["POST"])
def submit_time():
    """
    Receive actual time for a question and save for the user.
    Compare with expected time and add points if exceeded.
    """
    user_email = session.get("email")
    if not user_email:
        flash("You must be logged in to submit time.", "danger")
        return redirect(url_for("roadmap"))

    week = request.form.get("week")
    topic = request.form.get("topic")
    question = request.form.get("question")
    expected_time = int(request.form.get("expected_time", 0))
    actual_time = int(request.form.get("actual_time", 0))

    points = 0
    if actual_time > expected_time:
        # 1 point per minute over expected
        points = actual_time - expected_time

    # Update user's topic points in DB
    users.update_one(
        {"email": user_email},
        {"$inc": {f"topic_points.{topic}": points},
         "$set": {f"completed_questions.{question}": actual_time}},
        upsert=True
    )

    flash(f"Time for '{question}' saved! You earned {points} point(s) for topic '{topic}'.", "success")
    return redirect(url_for("roadmap"))

@app.route("/resume_checker", methods=["GET", "POST"])
def resume_checker():
    if request.method == "POST":
        role = request.form.get("role")
        company = request.form.get("company")
        file = request.files.get("resume")

        if not file or not allowed_file(file.filename):
            return render_template("resume_checker.html", error="Upload PDF/DOCX under 5 MB")

        filename = secure_filename(file.filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
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

            return render_template(
                "resume_checker.html",
                role=role,
                company=company,
                role_score=role_score,
                company_score=company_score,
                overall_score=overall_score
            )

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

from flask import session, flash, redirect, url_for, request, render_template
from random import shuffle
import logging

# Difficulty to expected minutes mapping
DIFFICULTY_TIME = {
    "easy": 15,
    "medium": 25,
    "hard": 40
}

@app.route("/roadmap")
def roadmap():
    """
    Render the roadmap page.
    Auto-generate roadmap if user has saved settings.
    """
    user_email = session.get("email")
    selected_topics = []
    weeks = None
    hours_per_week = None
    roadmap_data = None

    if user_email:
        user_data = users.find_one({"email": user_email})
        if user_data and "roadmap_settings" in user_data:
            settings = user_data["roadmap_settings"]
            weeks = settings.get("weeks")
            hours_per_week = settings.get("hours_per_week")
            selected_topics = settings.get("topics", [])

            # Fetch all problems
            all_problems = list(problems.find({"title": {"$ne": ""}}))
            if selected_topics:
                all_problems = [p for p in all_problems if infer_topic(p.get("title", "")) in selected_topics]

            if all_problems:
                shuffle(all_problems)
                minutes_per_week = hours_per_week * 60

                # Fetch user's completed questions
                completed = user_data.get("completed_questions", {})

                # Generate roadmap
                roadmap_data = {}
                problem_index = 0
                for week_num in range(1, weeks + 1):
                    week_tasks = []
                    used_minutes = 0
                    while used_minutes < minutes_per_week and problem_index < len(all_problems):
                        prob = all_problems[problem_index]
                        difficulty = prob.get("difficulty", "medium").lower()
                        expected_time = DIFFICULTY_TIME.get(difficulty, 20)
                        topic = infer_topic(prob.get("title", ""))

                        if used_minutes + expected_time <= minutes_per_week:
                            week_tasks.append({
                                "title": prob.get("title", "Unknown"),
                                "link": prob.get("link", "#"),
                                "difficulty": difficulty,
                                "expected_time_min": expected_time,
                                "topic": topic,
                                "completed": completed.get(prob.get("title", False))
                            })
                            used_minutes += expected_time
                            problem_index += 1
                        else:
                            break
                    roadmap_data[week_num] = week_tasks

    return render_template(
        "roadmap.html",
        show_sidebar=True,
        companies_list=TARGET_COMPANIES,
        topics_list=TOPICS_LIST,
        roadmap=roadmap_data,
        selected_topics=selected_topics,
        weeks=weeks,
        hours_per_week=hours_per_week
    )


@app.route("/generate_roadmap", methods=["POST"])
def generate_roadmap():
    """
    Generate roadmap based on user input.
    Persist the selections in DB per user email.
    """
    try:
        weeks = int(request.form.get("weeks", 0))
        hours_per_week = int(request.form.get("hours_per_week", 0))
        selected_topics = request.form.getlist("topics")

        if weeks <= 0 or hours_per_week <= 0:
            flash("Please fill all fields correctly.", "danger")
            return redirect(url_for("roadmap"))

        # Persist selections for the user
        user_email = session.get("email")
        if user_email:
            users.update_one(
                {"email": user_email},
                {"$set": {"roadmap_settings": {
                    "weeks": weeks,
                    "hours_per_week": hours_per_week,
                    "topics": selected_topics
                }}},
                upsert=True
            )

        # Fetch all problems
        all_problems = list(problems.find({"title": {"$ne": ""}}))

        # Filter by selected topics
        if selected_topics:
            all_problems = [
                p for p in all_problems if infer_topic(p.get("title", "")) in selected_topics
            ]

        if not all_problems:
            flash("No problems found for selected topics.", "warning")
            return redirect(url_for("roadmap"))

        shuffle(all_problems)
        minutes_per_week = hours_per_week * 60

        # Fetch user's completed questions
        completed = {}
        if user_email:
            user = users.find_one({"email": user_email})
            if user and "completed_questions" in user:
                completed = user["completed_questions"]

        # Generate roadmap
        roadmap = {}
        problem_index = 0
        for week_num in range(1, weeks + 1):
            week_tasks = []
            used_minutes = 0
            while used_minutes < minutes_per_week and problem_index < len(all_problems):
                prob = all_problems[problem_index]
                difficulty = prob.get("difficulty", "medium").lower()
                expected_time = DIFFICULTY_TIME.get(difficulty, 20)
                topic = infer_topic(prob.get("title", ""))

                if used_minutes + expected_time <= minutes_per_week:
                    week_tasks.append({
                        "title": prob.get("title", "Unknown"),
                        "link": prob.get("link", "#"),
                        "difficulty": difficulty,
                        "expected_time_min": expected_time,
                        "topic": topic,
                        "completed": completed.get(prob.get("title", ""), False)
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
            topics_list=TOPICS_LIST,
            roadmap=roadmap,
            selected_topics=selected_topics,
            weeks=weeks,
            hours_per_week=hours_per_week
        )

    except Exception as e:
        logging.error(f"Error generating roadmap: {e}")
        flash("Something went wrong while generating the roadmap.", "danger")
        return redirect(url_for("roadmap"))

@app.route("/toggle_completed", methods=["POST"])
def toggle_completed():
    if "email" not in session:
        return jsonify({"status": "error", "message": "User not logged in"}), 401

    user_email = session["email"]
    data = request.json
    question_title = data.get("title")

    if not question_title:
        return jsonify({"status": "error", "message": "No question title provided"}), 400

    # Fetch existing completed questions
    user = users.find_one({"email": user_email})
    completed = user.get("completed_questions", {}) if user else {}

    # Toggle completed status
    current_status = completed.get(question_title, False)
    completed[question_title] = not current_status

    # Update in DB
    users.update_one(
        {"email": user_email},
        {"$set": {"completed_questions": completed}},
        upsert=True
    )

    return jsonify({
        "status": "success",
        "title": question_title,
        "completed": completed[question_title]
    })


@app.route("/self_confidence")
def self_confidence():
    """
    Display user's self-confidence/performance summary grouped by topic.
    """
    user_email = session.get("email")
    if not user_email:
        flash("Please log in to see your self-confidence report.", "warning")
        return redirect(url_for("login"))

    user_data = users.find_one({"email": user_email})
    if not user_data:
        flash("No user data found.", "warning")
        return redirect(url_for("dashboard"))

    completed = user_data.get("completed_questions", {})  # { "question_title": seconds_taken }

    # Fetch all problems
    all_problems = list(problems.find({"title": {"$ne": ""}}))

    # Map difficulty to expected time and build topic-wise dict
    performance_by_topic = {}
    for prob in all_problems:
        title = prob.get("title")
        difficulty = prob.get("difficulty", "medium").lower()
        expected_time = DIFFICULTY_TIME.get(difficulty, 20)
        topic = infer_topic(title)
        actual_time = completed.get(title)

        # Decide status
        if actual_time is None:
            status = "Not Attempted"
        elif actual_time <= expected_time:
            status = "Completed On Time"
        else:
            status = "Exceeded Expected Time"

        entry = {
            "title": title,
            "difficulty": difficulty,
            "expected_time": expected_time,
            "actual_time": actual_time,
            "status": status,
            "link": prob.get("link", "#")
        }

        if topic not in performance_by_topic:
            performance_by_topic[topic] = []
        performance_by_topic[topic].append(entry)

    return render_template(
        "self_confidence.html",
        show_sidebar=True,
        performance_by_topic=performance_by_topic
    )

@app.route("/company_specific")
def company_specific():
    return render_template("company_specific.html", companies=TARGET_COMPANIES, show_sidebar=True)

# ------------------- CHATBOT -------------------
from google import genai

# Initialize Gemini client with API key
chatbot_client = genai.Client(api_key=GEMINI_API_KEY)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    prompt = f"""
    You are an AI Placement Guidance Chatbot for engineering students.
    User: "{user_message}"
    Provide guidance about placements, companies, skills, and interview tips.
    """

    try:
        response = chatbot_client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        reply = response.text.strip()
    except Exception as e:
        reply = f"Error: {str(e)}"

    return jsonify({"reply": reply})

@app.route("/company/<company_name>")
def company_questions(company_name):
    company_name = company_name.lower()
    company_problems = list(problems.find({"company_tags": company_name}, {"_id": 0}))
    return render_template(
        "company_questions.html",
        company=company_name,
        problems=company_problems,
        show_sidebar=True
    )

# ------------------- MAIN -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

