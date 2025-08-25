import streamlit as st
import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io
from streamlit_shap import st_shap

# 加载模型和SHAP解释器
model_1yr = joblib.load('./models/gbm_1yr.pkl')
model_3yr = joblib.load('./models/gbm_3yr.pkl')    
model_5yr = joblib.load('./models/gbm_5yr.pkl')

st.set_page_config(page_title="Clinical Decision Support System", layout="wide")
st.title("🩺 Clinical Decision Support System")
st.markdown(
    "This system predicts the **risk of kidney failure** using a machine learning model and explains the prediction with SHAP values."
)
#st.markdown("<hr>", unsafe_allow_html=True)
left_col, right_col = st.columns([2, 3], gap="large")

# 左侧特征输入
cakut_subphenotype_list = {
    'renal hypodysplasia associated with puv': 1,
    'solitary kidney': 2,
    'bilateral renal hypodysplasia': 3,
    'unilateral renal hypodysplasia': 4,
    'multicystic dysplastic kidney': 5,
    'horseshoe kidney': 6,
    'others': 7
}

with left_col:
    st.subheader("🏥 Patient Characteristics")
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        age_first_diagnose = st.number_input("Age At First Diagnose(yr)", min_value=0.0, max_value=18.0, value=0.0)
        gender = st.selectbox("Gender", ["Female", "Male"])
        family_history = st.selectbox("Family history", ["No", "Yes"])
        ckd_stage_first_diagnose = st.selectbox("CKD Stage At First Diagnose", [1, 2, 3, 4, 5])
        short_stature = st.selectbox("Short Stature", ["No", "Yes"])
        cakut_subphenotype = st.selectbox("CAKUT Subphenotype", cakut_subphenotype_list.keys())
        
    with col2:
        pax2 = st.selectbox("PAX2", ["No", "Yes"])
        prenatal_phenotype = st.selectbox("Prenatal Phenotype", ["No", "Yes"])
        congenital_heart_disease = st.selectbox("Congenital Heart Disease", ["No", "Yes"])
        ocular = st.selectbox("Ocular", ["No", "Yes"])
        preterm_birth = st.selectbox("Preterm Birth", ["No", "Yes"])
        behavioral_cognitive_abnormalities = st.selectbox("Behavioral Cognitive Abnormalities", ["No", "Yes"])

    predict_btn = st.button("PREDICT")
    
input_data = pd.DataFrame({
    'PAX2': [0 if pax2=='No' else 1],
    'age_first_diagnose': [age_first_diagnose],
    'behavioral_cognitive_abnormalities (1/0)': [0 if behavioral_cognitive_abnormalities=='No' else 1],
    'cakut_subphenotype': [cakut_subphenotype_list[cakut_subphenotype]],
    'ckd_stage_first_diagnose': [ckd_stage_first_diagnose],
    'congenital_heart_disease (1/0)': [0 if congenital_heart_disease=='No' else 1],
    'family_history (1/0)': [0 if family_history=='No' else 1],
    'gender (1/0)': [0 if gender == 'Female' else 1],
    'ocular (1/0)': [0 if ocular == 'No' else 1],
    'prenatal_phenotype (1/0)': [0 if prenatal_phenotype=='No' else 1],
    'preterm_birth (1/0)': [0 if preterm_birth=='No' else 1],
    'short_stature (1/0)': [0 if short_stature=='No' else 1]
})

def render_prediction(model, input_data, year):
    esrd = model.predict_proba(input_data)[0][1]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    st.write(f"Probability of kidney failure within {year} year: **{esrd:.2%}**")
    # html_buffer = io.StringIO()
    # force_plot_html = shap.force_plot(explainer.expected_value, shap_values[0], input_data, matplotlib=False)
    # shap.save_html(html_buffer, force_plot_html)
    # components.html(html_buffer.getvalue(), scrolling=True)
    
    shap.force_plot(explainer.expected_value, shap_values[0], input_data, matplotlib=True, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
 

    
with right_col:
    st.subheader("🤖 Predicted Results")
    if predict_btn:
        try:
            render_prediction(model_1yr, input_data, 1)
            render_prediction(model_3yr, input_data, 3)
            render_prediction(model_5yr, input_data, 5)
        except Exception as e:
            st.error(f"Error: {e}")
    
# # # 1. 模型预测
# # prob = model.predict_proba(user_input)[0][1]
# # st.subheader("📊 模型预测的患病概率")
# # st.metric(label="患病概率", value=f"{prob*100:.2f}%")

# # # 2. SHAP 分析
# # st.subheader("🧬 个体特征的重要性分析（SHAP）")
# # shap_values = explainer(user_input)

# # # 显示SHAP图（force_plot 或 bar_plot）
# # fig, ax = plt.subplots()
# # shap.plots.bar(shap_values[0], max_display=10, show=False)
# # st.pyplot(fig)
