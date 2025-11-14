import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
import contractions
import emoji

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
# print(df.shape)
# print(df.head(20))

# sarcasm_rows = df[df["class"] == "sarcasm"]
# print(sarcasm_rows)

train_df["label"] = train_df["class"].apply(lambda x: 1 if x == "sarcasm" else 0)
test_df["label"] = test_df["class"].apply(lambda x: 1 if x == "sarcasm" else 0)
# print(df)

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
# print(df["clean_text"])

X_train = train_df["clean_text"]
y_train = train_df["label"]
X_test = test_df["clean_text"]
y_test = test_df["label"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, 
#     y, 
#     test_size=0.2, 
#     random_state=42, 
#     stratify=df["label"]
# )

vectorizer = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(
        sublinear_tf=True,   # 1 + log(tf)
        ngram_range=(1,2),   # word unigrams + bigrams
        max_features=10000
    )),
    ('char_tfidf', TfidfVectorizer(
        analyzer='char_wb',  # character n-grams within word boundaries
        ngram_range=(3,5),   # trigrams to 5-grams
        max_features=5000
    ))
])

pipeline = Pipeline([
    ('features', vectorizer),
    ('clf', LogisticRegression(C=0.75, class_weight="balanced", max_iter=1000))
])

param_grid = {
    'features__word_tfidf__max_features': [5000, 10000],
    'features__char_tfidf__max_features': [2000, 5000],
    'clf__C': [0.5, 0.75, 1.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=3)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, probs)

# fpr, tpr, thresholds = roc_curve(y_test, probs)
# roc_auc = roc_auc_score(y_test, probs)

# plt.figure(figsize=(8,6))
# plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
# plt.plot([0,1], [0,1], color='gray', linestyle='--', label='Random baseline')

# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate (Recall)')
# plt.title('ROC Curve for Sarcasm Detection')
# plt.legend(loc='lower right')
# plt.savefig("log_reg_roc_curve.png", dpi=300, bbox_inches='tight')
# plt.show()

# custom_threshold = 0.47
# y_pred_custom = (probs >= custom_threshold).astype(int)

# plt.figure(figsize=(8,6))
# plt.plot(recall, precision, marker='.', label='Logistic Regression')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve for Sarcasm Detection')
# plt.legend()
# plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches='tight')
# # plt.show()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

f1_scores = 2 * (precision * recall) / (precision + recall)
f1_scores = np.nan_to_num(f1_scores)  # replace nan with 0

best_idx = f1_scores.argmax()
print("Best threshold:", thresholds[best_idx])
print("Best F1 score:", f1_scores[best_idx])

joblib.dump(best_model, "log_reg_sarcasm_model_v2.pkl")
print("Complete")
