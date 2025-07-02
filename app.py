import streamlit as st
import pickle

# Load model and vectorizer using your real filenames
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit page setup
st.set_page_config(page_title="SPAM MAIL DETECTOR", layout="centered")
st.title("📨 Spam vs Ham Mail Classifier")
st.markdown("Made by **Utkarsh  ❤️**")

# Message input
message = st.text_area("✉️ Enter your email or message text:")

# Prediction logic
if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        vector_input = vectorizer.transform([message])
        prediction = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]

        st.subheader("🔍 Result:")
        if prediction == 0:
            st.error(f"**Spam mail** detected!\n\nConfidence: {probability[0]*100:.2f}%")
        else:
            st.success(f"✅ **Ham mail** (Not Spam).\n\nConfidence: {probability[1]*100:.2f}%")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>App created with ❤️ by <b>Pandit Utkarsh</b></div>",
    unsafe_allow_html=True
)
