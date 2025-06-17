from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import pymysql
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib  # For saving model and scaler
from flask import send_from_directory
from sklearn.metrics import confusion_matrix, classification_report
import json
import numpy as np


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management


# File upload configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MySQL DB Config
db = pymysql.connect(host='localhost', user='root', password='Vision@09', database='userdb')
cursor = db.cursor()

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cursor.fetchone()
        if user:
            session['user'] = email
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for('home'))
    else:
        return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        name = request.form['name']
        email = request.form['email']
        gender = request.form['gender']
        dob = request.form['dob']
        password = request.form['password']
        phone = request.form['phone']
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            flash("Email already exists", "warning")
            return redirect(url_for('signup'))
        cursor.execute("INSERT INTO users (username,name, email,gender,dob, password, phone) VALUES (%s, %s, %s, %s, %s, %s, %s)", (username,name, email,gender,dob, password, phone))
        db.commit()
        flash("Signup successful. Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html', email=session['user'])

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'user' not in session:
        return redirect(url_for('home'))
    email = session['user']

    if 'dataset_file' not in request.files:
        flash("No file part", "warning")
        return redirect(url_for('dashboard'))

    file = request.files['dataset_file']
    if file.filename == '':
        flash("No file selected", "warning")
        return redirect(url_for('dashboard'))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Log upload info to DB
    cursor.execute("INSERT INTO uploads (email, filename) VALUES (%s, %s)", (email, file.filename))
    db.commit()

    try:
        df = pd.read_csv(filepath)
        df_preview = df.iloc[:10, :10]  # Show only top 10 rows and 10 columns
        preview_columns = df_preview.columns.tolist()
        dataset_preview = df_preview.values.tolist()
        flash("Dataset uploaded successfully!", "success")
        return render_template('dashboard.html', email=email, 
                               preview_columns=preview_columns, 
                               dataset_preview=dataset_preview,
                               filename=file.filename)
    except Exception as e:
        flash(f"Error reading dataset: {e}", "danger")
        return render_template('dashboard.html', email=email)

@app.route('/download_model/<filename>')
def download_model(filename):
    return send_from_directory('user', filename, as_attachment=True)

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('home'))

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if 'user' not in session:
        return redirect(url_for('login'))

    filename = request.form.get('filename')
    task_type = request.form.get('task_type')
    model_type = request.form.get('model_type')

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash("Dataset not found.", "danger")
        return redirect(url_for('dashboard'))

    # Load dataset
    df = pd.read_csv(filepath)

    # Basic assumption: last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert all columns in X to numeric (force strings to NaN)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Clean data: handle inf and NaN
    X.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    X.dropna(inplace=True)

    # Align y with cleaned X
    y = y.loc[X.index]

    # Encode labels for classification
    le = None
    if task_type == 'classification' and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    # Choose model
    if task_type == 'classification':
        if model_type == 'RandomForest':
            model = RandomForestClassifier()
        elif model_type == 'SVM':
            model = SVC()
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000)
        else:
            flash("Unsupported classification model selected.", "danger")
            return redirect(url_for('dashboard', filename=filename))
    else:
        flash("Only classification tasks are supported in this version.", "danger")
        return redirect(url_for('dashboard', filename=filename))

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    if task_type == 'classification':
        score = accuracy_score(y_test, y_pred)
    else:
        score = model.score(X_test, y_test)  # R^2, if regression

    # Save model
    os.makedirs('saved_models', exist_ok=True)
    model_path = os.path.join('saved_models', f"{filename}_{model_type}.pkl")
    joblib.dump(model, model_path)

    # Dataset preview (first 10 rows) after cleaning
    preview_X = X.head(10).to_dict(orient='records')
    # If y is a pandas Series, this works. If numpy array, convert to Series first.
    if isinstance(y, pd.Series):
        preview_y = y.head(10).tolist()
    else:
        preview_y = y[:10].tolist()


    # Save training log as JSON file
    now = datetime.now()
    log_data = {
        'email': session['user'],
        'filename': filename,
        'task_type': task_type,
        'model_type': model_type,
        'train_time': now.strftime("%Y-%m-%d %H:%M:%S"),
        'score': round(score, 4),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'class_labels': le.classes_.tolist() if le else None,
        'feature_names': X.columns.tolist(),
        'dataset_preview_X': preview_X,
        'dataset_preview_y': preview_y
    }

    os.makedirs('training_logs', exist_ok=True)
    log_filename = f"train_log_{session['user']}_{filename}_{model_type}.json"
    log_path = os.path.join('training_logs', log_filename)
    with open(log_path, 'w') as f:
        json.dump(log_data, f)

    # Store only the log filename in session
    session['train_log_filename'] = log_filename

    flash(f"Model trained successfully! Score: {score:.4f}", "success")
    return redirect(url_for('training_results'))


@app.route('/training_results')
def training_results():
    if 'user' not in session or 'train_log_filename' not in session:
        flash("No training results found.", "warning")
        return redirect(url_for('dashboard'))

    log_filename = session['train_log_filename']
    log_path = os.path.join('training_logs', log_filename)

    if not os.path.exists(log_path):
        flash("Training log file missing.", "danger")
        return redirect(url_for('dashboard'))

    with open(log_path, 'r') as f:
        log_data = json.load(f)

    # Extract info from log
    score = log_data['score']
    feature_names = log_data['feature_names']
    dataset_preview_X = log_data['dataset_preview_X']
    dataset_preview_y = log_data['dataset_preview_y']
    y_test = np.array(log_data['y_test'])
    y_pred = np.array(log_data['y_pred'])
    class_labels = log_data.get('class_labels')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report (dict)
    report = classification_report(y_test, y_pred, target_names=class_labels, zero_division=0, output_dict=True)

    return render_template(
        'training_results.html',
        score=score,
        feature_names=feature_names,
        dataset_preview_X=dataset_preview_X,
        dataset_preview_y=dataset_preview_y,
        confusion_matrix=cm.tolist(),
        classification_report=report,
        class_labels=class_labels
    )

@app.route('/train_log', methods=['GET'])
def train_log():
    if 'user' not in session or 'train_log_filename' not in session:
        flash("No training log found.", "warning")
        return redirect(url_for('dashboard'))

    log_path = os.path.join('training_logs', session['train_log_filename'])
    if not os.path.exists(log_path):
        flash("Training log file missing.", "danger")
        return redirect(url_for('dashboard'))

    with open(log_path, 'r') as f:
        log = json.load(f)

    return render_template('train_log.html', log=[
        log['email'],
        log['filename'],
        log['task_type'],
        log['model_type'],
        log['train_time'],
        log['score']
    ])

@app.route('/confusion_matrix/<filename>/<model_type>')
def confusion_matrix_view(filename, model_type):
    if 'user' not in session or 'train_log_filename' not in session:
        flash("No training data found.", "warning")
        return redirect(url_for('dashboard'))

    log_path = os.path.join('training_logs', session['train_log_filename'])
    if not os.path.exists(log_path):
        flash("Training log file missing.", "danger")
        return redirect(url_for('dashboard'))

    with open(log_path, 'r') as f:
        log = json.load(f)

    if log['filename'] != filename or log['model_type'] != model_type:
        flash("Requested model or file does not match last training.", "warning")
        return redirect(url_for('dashboard'))

    if log['task_type'] != 'classification':
        flash("Confusion matrix available only for classification task.", "warning")
        return redirect(url_for('dashboard'))

    y_test = log['y_test']
    y_pred = log['y_pred']
    class_labels = log.get('class_labels')

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_labels, zero_division=0, output_dict=True)

    return render_template(
        'confusion_matrix.html',
        cm=cm,
        report=report,
        labels=class_labels
    )

if __name__ == '__main__':
    app.run(debug=True)
