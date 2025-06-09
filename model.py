import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib

# --- Load Data ---
df = pd.read_csv('./Medicaldataset.csv')
x = df.drop('Result', axis=1)
y = df['Result'].map({'negative': 0, 'positive': 1})
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Load Models ---
model1 = joblib.load('model1_xgb.pkl')
model2 = joblib.load('model2_rf.pkl')
model3 = joblib.load('model3_knn.pkl')
model4 = joblib.load('model4_dt.pkl')
model5 = joblib.load('model5_gb.pkl')
model6 = joblib.load('model6_lr.pkl')

models = [model1, model2, model3, model4, model5, model6]
model_names = ['XGBoost', 'Random Forest', 'KNN', 'Decision Tree', 'Gradient Boosting', 'Logistic Regression']

# --- UI: Title ---
st.title("💓 Heart Attack Prediction Models Comparison")

# --- Sidebar: User Prediction Form ---
st.sidebar.header("🧠 Predict with Your Own Input")
user_input = {}
for col in x.columns:
    user_input[col] = st.sidebar.number_input(f"{col}", value=float(x[col].mean()), key=col)

st.sidebar.write("Input preview:", user_input)
input_df = pd.DataFrame([user_input])


# Model selection using model names (strings)
pred_model_name = st.sidebar.selectbox(
    "Choose model for prediction", 
    model_names,
    key="predict_model_selectbox"
)

# Predict button
if st.sidebar.button("🧾 Predict"):
    pred_model = models[model_names.index(pred_model_name)]
    prediction = pred_model.predict(input_df)[0]
    result = "🟥 Positive (At Risk)" if prediction == 1 else "🟩 Negative (Low Risk)"
    
    st.subheader("🔎 Prediction Result")
    if hasattr(pred_model, "predict_proba"):
        prob = pred_model.predict_proba(input_df)[0, 1]
        st.success(f"**{result}**\n\n📈 Probability of Risk: `{prob:.2f}`")
    else:
        st.success(f"**{result}**")
        st.warning("Probability score not supported for this model.")
    
    # Download result CSV
    result_df = input_df.copy()
    result_df['Prediction'] = result
    if hasattr(pred_model, "predict_proba"):
        result_df['Prediction Probability'] = prob
    csv = result_df.to_csv(index=False)
    st.download_button("⬇️ Download Prediction", csv, "prediction_result.csv", "text/csv")

# --- Model Evaluation Metrics ---
metrics_data = []
for name, model in zip(model_names, models):
    preds = model.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    metrics_data.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall (Sensitivity)': recall_score(y_test, preds),
        'Specificity': tn / (tn + fp),
        'F1 Score': f1_score(y_test, preds)
    })

if metrics_data:
    df_metrics = pd.DataFrame(metrics_data)
    st.subheader("📊 Model Performance Metrics")
    st.dataframe(df_metrics.style.format({col: "{:.2f}" for col in df_metrics.select_dtypes(include='number')}))
else:
    st.info("Please select at least one model to compare.")

# --- ROC Curve Plotting ---
def plot_roc_curves(models, names, x_test, y_test):
    plt.figure(figsize=(8, 6))
    for model, name in zip(models, names):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("🧪 ROC Curve Comparison")
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
    plt.clf()

plot_roc_curves(models, model_names, x_test, y_test)
