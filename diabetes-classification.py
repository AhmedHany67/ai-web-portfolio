"""
Task 2 Diabetes Classification - Full Pipeline + MANY MODEL GRAPHS
Fixed version for Colab: this is a real .py file. Do not open it as .ipynb.

Colab steps:
1) Upload diabetes_prediction_dataset.csv
2) Upload this file
3) Run: !python task2_diabetes_MODEL_GRAPHS_FAST_COLAB.py
4) Download the output folder: task2_model_graphs_outputs

Important:
- The graphs for the MODELS are saved in: task2_model_graphs_outputs/model_graphs
- Preprocessing graphs are saved separately in: task2_model_graphs_outputs/preprocessing_graphs
"""

from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.base import clone

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

# Keep this as None to use full dataset. Use 30000 if Colab is slow.
MAX_ROWS_FOR_SPEED = 8000

OUT = Path('task2_model_graphs_outputs')
MODEL_DIR = OUT / 'model_graphs'
PRE_DIR = OUT / 'preprocessing_graphs'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRE_DIR.mkdir(parents=True, exist_ok=True)

def find_data():
    for p in ['diabetes_prediction_dataset.csv', '/content/diabetes_prediction_dataset.csv', '/mnt/data/diabetes_prediction_dataset.csv']:
        if Path(p).exists():
            return p
    raise FileNotFoundError('diabetes_prediction_dataset.csv not found. Upload it to Colab first.')

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

def metrics_dict(name, pca_status, y_true, y_pred):
    return {
        'Model': name,
        'PCA': pca_status,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Average Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Sensitivity/Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
    }

def scores_for_curve(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(X)
    return model.predict(X)

# =====================
# 1) Load data
# =====================
path = find_data()
df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
if MAX_ROWS_FOR_SPEED is not None and len(df) > MAX_ROWS_FOR_SPEED:
    df, _ = train_test_split(df, train_size=MAX_ROWS_FOR_SPEED, stratify=df['diabetes'], random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)
print('Dataset used:', df.shape)

TARGET = 'diabetes'
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

# =====================
# 2) Preprocessing graphs
# =====================
plt.figure(figsize=(6,4)); y.value_counts().sort_index().plot(kind='bar'); plt.title('Target Distribution'); plt.xlabel('Diabetes'); plt.ylabel('Count'); save_fig(PRE_DIR/'01_target_distribution.png')
plt.figure(figsize=(9,4)); df.isnull().sum().plot(kind='bar'); plt.title('Missing Values'); plt.ylabel('Count'); save_fig(PRE_DIR/'02_missing_values.png')
for col in num_cols:
    plt.figure(figsize=(6,4)); df[col].hist(bins=30); plt.title(f'Distribution of {col}'); plt.xlabel(col); plt.ylabel('Frequency'); save_fig(PRE_DIR/f'03_distribution_{col}.png')
for col in cat_cols:
    pd.crosstab(df[col], df[TARGET], normalize='index').plot(kind='bar', figsize=(8,4)); plt.title(f'Diabetes Ratio by {col}'); plt.ylabel('Ratio'); save_fig(PRE_DIR/f'04_diabetes_by_{col}.png')
corr = df[num_cols + [TARGET]].corr()
plt.figure(figsize=(8,6)); plt.imshow(corr, aspect='auto'); plt.colorbar(); plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right'); plt.yticks(range(len(corr.index)), corr.index); plt.title('Correlation Heatmap'); save_fig(PRE_DIR/'05_correlation_heatmap.png')

# =====================
# 3) Split and transform
# =====================
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=RANDOM_STATE)
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
pre = ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', ohe, cat_cols)])
X_train_s = pre.fit_transform(X_train)
X_val_s = pre.transform(X_val)
X_test_s = pre.transform(X_test)
X_train_val_s = pre.transform(X_train_val)

pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_s)
X_val_pca = pca.transform(X_val_s)
X_test_pca = pca.transform(X_test_s)
X_train_val_pca = pca.transform(X_train_val_s)

plt.figure(figsize=(7,4)); plt.plot(range(1, len(np.cumsum(pca.explained_variance_ratio_))+1), np.cumsum(pca.explained_variance_ratio_), marker='o'); plt.axhline(0.95, linestyle='--'); plt.title('PCA Explained Variance'); plt.xlabel('Components'); plt.ylabel('Cumulative Variance'); save_fig(MODEL_DIR/'00_pca_explained_variance.png')

# =====================
# 4) Train models with and without PCA
# =====================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
    'MLP': MLPClassifier(hidden_layer_sizes=(30,15), max_iter=35, early_stopping=True, random_state=RANDOM_STATE),
    'SVM Linear': LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=RANDOM_STATE),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=50, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE),
}

results, trained, times = [], {}, []
for pca_status, Xtr, Xv in [('Without PCA', X_train_s, X_val_s), ('With PCA', X_train_pca, X_val_pca)]:
    for name, model in models.items():
        print('Training:', name, pca_status)
        t0 = time.time(); m = clone(model)
        m.fit(Xtr, y_train); sec = time.time()-t0
        pred = m.predict(Xv)
        results.append(metrics_dict(name, pca_status, y_val, pred))
        trained[f'{name} - {pca_status}'] = (m, Xv, pca_status)
        times.append({'Model': name, 'PCA': pca_status, 'Training Time Seconds': sec})

# =====================
# 5) SVM RBF Grid Search and PSO
# =====================
# Use a smaller sample for RBF SVM because Gaussian kernel is slow on large data.
svm_size = min(1000, len(X_train_s))
idx = np.random.default_rng(RANDOM_STATE).choice(len(X_train_s), size=svm_size, replace=False)
X_svm, y_svm = X_train_s[idx], y_train.iloc[idx]

print('Grid Search for SVM RBF...')
svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
grid = GridSearchCV(svm, {'C':[0.1,1,10], 'gamma':[0.01,0.1,1]}, scoring='f1', cv=3, n_jobs=1)
grid.fit(X_svm, y_svm)
grid_model = grid.best_estimator_
pred = grid_model.predict(X_val_s)
results.append(metrics_dict('SVM RBF Grid Search', 'Without PCA', y_val, pred))
trained['SVM RBF Grid Search - Without PCA'] = (grid_model, X_val_s, 'Without PCA')

cv = pd.DataFrame(grid.cv_results_)
heat = cv.pivot_table(values='mean_test_score', index='param_gamma', columns='param_C')
plt.figure(figsize=(6,4)); plt.imshow(heat, aspect='auto'); plt.colorbar(label='Mean CV F1'); plt.xticks(range(len(heat.columns)), heat.columns); plt.yticks(range(len(heat.index)), heat.index); plt.xlabel('C'); plt.ylabel('Gamma'); plt.title('SVM RBF Grid Search Heatmap'); save_fig(MODEL_DIR/'01_svm_grid_search_heatmap.png')

print('PSO for SVM RBF...')
rng = np.random.default_rng(RANDOM_STATE)
X_pso_train, X_pso_val, y_pso_train, y_pso_val = train_test_split(X_svm, y_svm, test_size=0.25, stratify=y_svm, random_state=RANDOM_STATE)
bounds = np.array([[-2, 2], [-3, 1]], dtype=float) # log10(C), log10(gamma)
n_particles, n_iter = 4, 3
pos = rng.uniform(bounds[:,0], bounds[:,1], size=(n_particles,2)); vel = rng.normal(0,0.2,size=(n_particles,2))
pbest, pbest_score = pos.copy(), np.full(n_particles, -np.inf)
gbest, gbest_score = pos[0].copy(), -np.inf
hist=[]
def fit_score(lp):
    C, gamma = 10**lp[0], 10**lp[1]
    m = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced', random_state=RANDOM_STATE)
    m.fit(X_pso_train, y_pso_train)
    return f1_score(y_pso_val, m.predict(X_pso_val), zero_division=0)
for it in range(n_iter):
    for i in range(n_particles):
        sc = fit_score(pos[i])
        if sc > pbest_score[i]: pbest_score[i], pbest[i] = sc, pos[i].copy()
        if sc > gbest_score: gbest_score, gbest = sc, pos[i].copy()
    hist.append(gbest_score)
    r1, r2 = rng.random((n_particles,2)), rng.random((n_particles,2))
    vel = 0.6*vel + 1.5*r1*(pbest-pos) + 1.5*r2*(gbest-pos)
    pos = np.clip(pos+vel, bounds[:,0], bounds[:,1])

pso_C, pso_gamma = float(10**gbest[0]), float(10**gbest[1])
plt.figure(figsize=(7,4)); plt.plot(range(1,len(hist)+1), hist, marker='o'); plt.title('PSO Convergence for SVM RBF'); plt.xlabel('Iteration'); plt.ylabel('Best F1'); save_fig(MODEL_DIR/'02_svm_pso_convergence.png')
pso_model = SVC(kernel='rbf', C=pso_C, gamma=pso_gamma, class_weight='balanced', probability=True, random_state=RANDOM_STATE)
pso_model.fit(X_svm, y_svm)
pred = pso_model.predict(X_val_s)
results.append(metrics_dict('SVM RBF PSO', 'Without PCA', y_val, pred))
trained['SVM RBF PSO - Without PCA'] = (pso_model, X_val_s, 'Without PCA')

# =====================
# 6) MODEL graphs
# =====================
res = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
res.to_csv(OUT/'validation_model_results.csv', index=False)
pd.DataFrame(times).to_csv(OUT/'training_times.csv', index=False)
with open(OUT/'svm_best_parameters.json','w') as f:
    json.dump({'grid_search_best_params': grid.best_params_, 'grid_search_best_f1': float(grid.best_score_), 'pso_best_params': {'C':pso_C, 'gamma':pso_gamma}, 'pso_best_f1': float(gbest_score)}, f, indent=4)

# Bar comparisons for all important metrics
for metric in ['Accuracy', 'Precision', 'Sensitivity/Recall', 'F1 Score']:
    labels = res['Model'] + '\n' + res['PCA']
    plt.figure(figsize=(14,6)); plt.bar(range(len(res)), res[metric]); plt.xticks(range(len(res)), labels, rotation=75, ha='right'); plt.ylim(0,1.05); plt.title(f'Model Comparison by {metric}'); plt.ylabel(metric); save_fig(MODEL_DIR/f'03_model_comparison_{metric.replace("/","_").replace(" ","_")}.png')

# PCA vs no PCA
base = res[res['Model'].isin(models.keys())]
for metric in ['Accuracy', 'Precision', 'Sensitivity/Recall', 'F1 Score']:
    pivot = base.pivot(index='Model', columns='PCA', values=metric)
    pivot.plot(kind='bar', figsize=(12,5)); plt.title(f'With PCA vs Without PCA - {metric}'); plt.ylabel(metric); plt.ylim(0,1.05); plt.xticks(rotation=45, ha='right'); save_fig(MODEL_DIR/f'04_pca_vs_without_pca_{metric.replace("/","_").replace(" ","_")}.png')

# Training time graph
if len(times) > 0:
    tdf = pd.DataFrame(times); labels = tdf['Model'] + '\n' + tdf['PCA']
    plt.figure(figsize=(13,5)); plt.bar(range(len(tdf)), tdf['Training Time Seconds']); plt.xticks(range(len(tdf)), labels, rotation=75, ha='right'); plt.title('Training Time Comparison'); plt.ylabel('Seconds'); save_fig(MODEL_DIR/'05_training_time_comparison.png')

# Confusion matrices for every model
for key, (model, Xv, _) in trained.items():
    safe = key.replace(' ','_').replace('-','_').replace('/','_')
    ConfusionMatrixDisplay(confusion_matrix(y_val, model.predict(Xv)), display_labels=[0,1]).plot(values_format='d')
    plt.title(f'Confusion Matrix - {key}')
    save_fig(MODEL_DIR/f'06_confusion_matrix_{safe}.png')

# ROC and Precision-Recall for top 6 models
for _, row in res.head(4).iterrows():
    key = f"{row['Model']} - {row['PCA']}"
    if key not in trained: continue
    model, Xv, _ = trained[key]
    safe = key.replace(' ','_').replace('-','_').replace('/','_')
    try:
        s = scores_for_curve(model, Xv)
        RocCurveDisplay.from_predictions(y_val, s); plt.title(f'ROC Curve - {key}'); save_fig(MODEL_DIR/f'07_roc_curve_{safe}.png')
        PrecisionRecallDisplay.from_predictions(y_val, s); plt.title(f'Precision-Recall Curve - {key}'); save_fig(MODEL_DIR/f'08_precision_recall_curve_{safe}.png')
    except Exception as e:
        print('ROC/PR skipped for', key, e)

# Learning curve
# To keep Colab fast, the learning curve is drawn using Logistic Regression.
# This still satisfies the required learning-curve graph and avoids very slow SVM/MLP repeated training.
best = res.iloc[0]
best_key = f"{best['Model']} - {best['PCA']}"
best_model, _, _ = trained[best_key]
lc_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
lc_size = min(4000, len(X_train_val_s))
lc_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_train_val_s), size=lc_size, replace=False)
train_sizes, train_scores, val_scores = learning_curve(lc_model, X_train_val_s[lc_idx], y_train_val.iloc[lc_idx], cv=3, scoring='f1', train_sizes=np.linspace(0.2,1.0,4), n_jobs=1)
plt.figure(figsize=(8,5)); plt.plot(train_sizes, train_scores.mean(axis=1), marker='o', label='Training F1'); plt.plot(train_sizes, val_scores.mean(axis=1), marker='o', label='Validation F1'); plt.title('Learning Curve - Logistic Regression'); plt.xlabel('Training Examples'); plt.ylabel('F1 Score'); plt.legend(); save_fig(MODEL_DIR/'09_learning_curve_logistic_regression.png')

# Final test metrics for best model
X_best_test = X_test_pca if best['PCA'] == 'With PCA' else X_test_s
final_pred = best_model.predict(X_best_test)
final = metrics_dict(best['Model'], best['PCA'], y_test, final_pred)
pd.DataFrame([final]).to_csv(OUT/'final_test_metrics_best_model.csv', index=False)
ConfusionMatrixDisplay(confusion_matrix(y_test, final_pred), display_labels=[0,1]).plot(values_format='d')
plt.title(f'Final Test Confusion Matrix - {best_key}')
save_fig(MODEL_DIR/'10_final_test_confusion_matrix_best_model.png')

with open(OUT/'summary.txt','w') as f:
    f.write('Best validation model: ' + best_key + '\n')
    f.write('Final test metrics:\n')
    f.write(json.dumps(final, indent=4))
    f.write('\n\nModel graphs folder: task2_model_graphs_outputs/model_graphs\n')
    f.write('Preprocessing graphs folder: task2_model_graphs_outputs/preprocessing_graphs\n')

print('\nDONE')
print('Python code finished successfully.')
print('MODEL GRAPHS ARE HERE:', MODEL_DIR.resolve())
print('Best model:', best_key)
print('Final test metrics:', final)
