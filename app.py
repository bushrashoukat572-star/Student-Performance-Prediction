# v2import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#2ecc71;'>🎓 AI-Powered Student Performance Prediction</h1>
    <p style='text-align:center; color:gray;'>Name: <b>Bushra Shoukat</b> &nbsp;|&nbsp; Internship ID: <b>ZYNVEX-CERT-1947</b></p>
    <hr>
""", unsafe_allow_html=True)

# ── Generate & Train Model ───────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500
    attendance  = np.random.uniform(40, 100, n)
    study_hours = np.random.uniform(1, 10, n)
    prev_marks  = np.random.uniform(30, 100, n)
    test_scores = np.random.uniform(30, 100, n)

    avg_score = (attendance * 0.20 + prev_marks * 0.40 +
                 test_scores * 0.30 + study_hours * 5 * 0.10)

    def label(s):
        if s >= 70: return 'Good'
        elif s >= 50: return 'Average'
        else: return 'Low'

    performance = [label(s) for s in avg_score]

    df = pd.DataFrame({
        'attendance_pct': attendance.round(1),
        'study_hours_per_day': study_hours.round(1),
        'previous_marks': prev_marks.round(1),
        'test_scores': test_scores.round(1),
        'performance': performance
    })

    X = df[['attendance_pct','study_hours_per_day','previous_marks','test_scores']]
    y = df['performance']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler, le, df, X_test, y_test

model, scaler, le, df, X_test, y_test = train_model()

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Analysis", "📋 Model Report"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Student Details")
    col1, col2 = st.columns(2)
    with col1:
        attendance   = st.slider("📅 Attendance (%)", 0, 100, 75)
        study_hours  = st.slider("📚 Study Hours / Day", 0.0, 12.0, 5.0, 0.5)
    with col2:
        prev_marks   = st.number_input("📝 Previous Marks (0–100)", 0, 100, 70)
        test_score   = st.number_input("🧪 Test Score (0–100)", 0, 100, 72)

    if st.button("🔍 Predict Performance", use_container_width=True):
        features = np.array([[attendance, study_hours, prev_marks, test_score]])
        f_sc     = scaler.transform(features)
        pred     = model.predict(f_sc)[0]
        proba    = model.predict_proba(f_sc)[0]
        perf     = le.inverse_transform([pred])[0]
        est_score = round(test_score * 0.6 + prev_marks * 0.4, 2)
        conf     = round(max(proba) * 100, 1)

        color = {'Good':'#2ecc71','Average':'#f39c12','Low':'#e74c3c'}
        emoji = {'Good':'🟢','Average':'🟡','Low':'🔴'}
        advice= {'Good':'✅ Excellent! Keep it up.',
                 'Average':'📚 Improve study hours & attendance.',
                 'Low':'⚠️ Serious attention required!'}

        st.markdown(f"""
        <div style='background:{color[perf]}22; border-left:5px solid {color[perf]};
                    padding:20px; border-radius:10px; margin-top:20px;'>
            <h2 style='color:{color[perf]};'>{emoji[perf]} Predicted Performance: {perf}</h2>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Estimated Score", f"{est_score}/100")
        c2.metric("💯 Confidence",       f"{conf}%")
        c3.metric("🎯 Performance",      perf)

        st.info(advice[perf])

# ════════════════════════════════════════════════════════════════
# TAB 2 — DATA ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df))
    col2.metric("Good",    int((df['performance']=='Good').sum()))
    col3.metric("Average", int((df['performance']=='Average').sum()))

    colors = {'Good':'#2ecc71','Average':'#f39c12','Low':'#e74c3c'}
    order  = ['Good','Average','Low']

    fig, axes = plt.subplots(2, 3, figsize=(16,10))
    fig.suptitle('Student Performance Analysis', fontsize=16, fontweight='bold')

    # Pie
    counts = df['performance'].value_counts()
    axes[0,0].pie(counts, labels=counts.index, autopct='%1.1f%%',
                  colors=[colors[k] for k in counts.index],
                  wedgeprops={'edgecolor':'white','linewidth':2})
    axes[0,0].set_title('Performance Distribution', fontweight='bold')

    # Box
    data_box = [df[df['performance']==p]['attendance_pct'].values for p in order]
    bp = axes[0,1].boxplot(data_box, tick_labels=order, patch_artist=True)
    for patch, p in zip(bp['boxes'], order):
        patch.set_facecolor(colors[p])
    axes[0,1].set_title('Attendance by Performance', fontweight='bold')
    axes[0,1].grid(axis='y', alpha=0.3)

    # Histogram
    for p in order:
        axes[0,2].hist(df[df['performance']==p]['study_hours_per_day'],
                       alpha=0.65, label=p, color=colors[p], bins=15, edgecolor='white')
    axes[0,2].set_title('Study Hours Distribution', fontweight='bold')
    axes[0,2].legend()
    axes[0,2].grid(alpha=0.3)

    # Scatter
    for p in order:
        sub = df[df['performance']==p]
        axes[1,0].scatter(sub['previous_marks'], sub['test_scores'],
                          alpha=0.5, label=p, color=colors[p], s=25)
    axes[1,0].set_title('Previous Marks vs Test Scores', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    # Heatmap
    sns.heatmap(df.select_dtypes(include=np.number).corr(),
                annot=True, fmt='.2f', cmap='RdYlGn',
                ax=axes[1,1], linewidths=0.5)
    axes[1,1].set_title('Correlation Heatmap', fontweight='bold')

    # Bar
    avg = df.groupby('performance')[['previous_marks','test_scores']].mean().loc[order]
    x = np.arange(3); w = 0.35
    axes[1,2].bar(x-w/2, avg['previous_marks'], w, label='Prev Marks',
                  color='#3498db', edgecolor='white')
    axes[1,2].bar(x+w/2, avg['test_scores'],    w, label='Test Scores',
                  color='#9b59b6', edgecolor='white')
    axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(order)
    axes[1,2].set_title('Avg Scores by Performance', fontweight='bold')
    axes[1,2].legend(); axes[1,2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL REPORT
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Report")

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    c1, c2 = st.columns(2)
    c1.metric("🎯 Model Accuracy", f"{acc*100:.2f}%")
    c2.metric("🌲 Algorithm", "Random Forest")

    st.text("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    st.code(report)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12,5))

    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=axes2[0], cmap='Blues', colorbar=False)
    axes2[0].set_title('Confusion Matrix', fontweight='bold')

    imp  = model.feature_importances_
    feat = ['Attendance','Study Hours','Prev Marks','Test Scores']
    idx  = np.argsort(imp)[::-1]
    axes2[1].bar(range(4), imp[idx],
                 color=['#e74c3c','#e67e22','#3498db','#2ecc71'],
                 edgecolor='white')
    axes2[1].set_xticks(range(4))
    axes2[1].set_xticklabels([feat[i] for i in idx])
    axes2[1].set_title('Feature Importance', fontweight='bold')
    axes2[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Bushra Shoukat | ZYNVEX-CERT-1947 | AI/ML Internship Project</p>",
            unsafe_allow_html=True)
        
    
