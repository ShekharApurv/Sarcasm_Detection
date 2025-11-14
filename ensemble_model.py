import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import contractions
import emoji
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import seaborn as sns

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df["label"] = train_df["class"].apply(lambda x: 1 if x == "sarcasm" else 0)
test_df["label"] = test_df["class"].apply(lambda x: 1 if x == "sarcasm" else 0)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"#\w+", "", text)          # remove hashtags
    text = contractions.fix(text)         # expand contractions
    text = emoji.replace_emoji(text, replace="") # remove emoji
    text = re.sub(r"[^\w\s]", "", text)    # remove punctuation
    text = text.lower().strip()
    return text

train_df["clean_text"] = train_df["tweets"].apply(clean_text)
test_df["tweets"] = test_df["tweets"].astype(str)
test_df["clean_text"] = test_df["tweets"].apply(clean_text)

X_train = train_df["clean_text"]
y_train = train_df["label"]
X_test = test_df["clean_text"]
y_test = test_df["label"]

# Base models
log_reg = LogisticRegression(
    C=1.0, 
    class_weight="balanced", 
    max_iter=1000
)
svm = CalibratedClassifierCV(
    cv=5,
    estimator=LinearSVC(
        class_weight="balanced", 
        max_iter=5000
    ),
    method="sigmoid"
)  # calibration for probabilities
rf = RandomForestClassifier(
    n_estimators=200, 
    class_weight="balanced", 
    random_state=42
)

# Pipeline with TF-IDF + ensemble
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        sublinear_tf=True, 
        ngram_range=(1,2), 
        max_features=10000
    )),
    ('ensemble', VotingClassifier(
        estimators=[('lr', log_reg), ('svm', svm), ('rf', rf)],
        voting='soft'  # use probabilities for averaging
    ))
])

param_grid = {
    'tfidf__max_features': [5000, 10000],
    'ensemble__lr__C': [0.5, 1.0, 2.0],   # tune Logistic Regression
    'ensemble__svm__estimator__C': [0.5, 0.75, 1.0],  # tune SVM
    'ensemble__rf__n_estimators': [100, 200]
}

grid = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='f1_macro', 
    n_jobs=-1, 
    verbose=3
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best F1:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# fpr, tpr, thresholds = roc_curve(y_test, probs)
# roc_auc = roc_auc_score(y_test, probs)

# plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
# plt.plot([0,1], [0,1], linestyle='--', color='gray')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()

# precision, recall, thresholds = precision_recall_curve(y_test, probs)
# plt.plot(recall, precision, marker='.')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision–Recall Curve')
# plt.show()

joblib.dump(best_model, "ensemble_sarcasm_model_v1.pkl")