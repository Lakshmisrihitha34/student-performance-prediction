from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import pandas as pd
import numpy as np
import pickle
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "secret123"

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- DATABASE SETUP ----------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT,
        password TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        knowledge_test REAL,
        study_hours REAL,
        previous_marks REAL,
        assignments REAL,
        prediction TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

init_db()

# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return redirect(url_for("login"))

# ---------- REGISTER ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        try:
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                      (username, email, password))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Try another.", "danger")
    return render_template("register.html")

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[3], password):
            session["username"] = username
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")

# ---------- HOME ----------
@app.route("/home")
def home():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("home.html", username=session["username"])

# ---------- PREDICTION ----------
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "username" not in session:
        return redirect(url_for("login"))

    result = None
    if request.method == "POST":
        kt = float(request.form["knowledge_test"])
        sh = float(request.form["study_hours"])
        pm = float(request.form["previous_marks"])
        asg = float(request.form["assignments"])

        features = np.array([[kt, sh, pm, asg]])
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        pred_label = "PASS" if pred == 1 else "FAIL"

        # Save to DB
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("""INSERT INTO predictions
                     (username, knowledge_test, study_hours, previous_marks, assignments, prediction)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (session["username"], kt, sh, pm, asg, pred_label))
        conn.commit()
        conn.close()

        # Generate recommendations
        tips = []
        if kt < 70:
            tips.append("Improve knowledge test performance through regular practice.")
        if sh < 3:
            tips.append("Increase daily study hours to at least 3-4 hours.")
        if pm < 60:
            tips.append("Revise previous academic concepts.")
        if asg < 75:
            tips.append("Complete assignments regularly and on time.")
        if pred_label == "PASS":
            tips.append("Great! Maintain your current study habits.")
        else:
            tips.append("Focus on weak areas and seek academic support.")

        result = {
            "prediction": pred_label,
            "recommendation": tips,
            "input": {"kt": kt, "sh": sh, "pm": pm, "asg": asg}
        }

    return render_template("prediction.html", result=result)

# ---------- ABOUT ----------
@app.route("/about")
def about():
    return render_template("about.html")

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)