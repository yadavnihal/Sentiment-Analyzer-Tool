import streamlit as st
import pandas as pd
import spacy
import joblib
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

nlp = spacy.load("en_core_web_sm")

def lemmatization(text):
    doc = nlp(text)
    lemmaList = [word.lemma_ for word in doc]
    return ' '.join(lemmaList)

def spacy_tokenizer(text):
    text = lemmatization(text)
    doc = nlp(text)
    return [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]

st.set_page_config(page_title="Sentiment Analyzer Tool", page_icon="💬", layout="centered")

st.sidebar.header("ℹ️ How to Use")
st.sidebar.markdown("""
- ✍️ Enter a review in the text area  
- 📁 Or upload a CSV file with a 'text' column  
- 📊 View the sentiment prediction  
- 💾 Model: Logistic Regression with TF-IDF & SpaCy
""")
st.sidebar.markdown("---")
st.sidebar.write("🔁 Developed by Nihal Yadav")

st.markdown("""
    <div style="text-align:center">
        <h1>🧠 Sentiment Analyzer</h1>
        <p style="font-size:18px;">Analyze customer feedback, reviews, or any text data!</p>
    </div>
""", unsafe_allow_html=True)

st.subheader("🔍 Analyze Single Text")
with st.form("text_form"):
    user_input = st.text_area("Enter your review here:")
    submit = st.form_submit_button("Analyze")

if submit:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        input_vec = tfidf.transform([user_input])
        prediction = model.predict(input_vec)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        color = "green" if prediction == 1 else "red"
        st.markdown(f"<h3 style='color:{color}'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)

st.markdown("---")

st.subheader("📁 Analyze Multiple Reviews (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("❌ The CSV must contain a column named 'text'.")
        else:
            if st.button("🔍 Analyze File"):
                with st.spinner("Analyzing..."):
                    df['text'] = df['text'].astype(str)
                    df['lemmatized'] = df['text'].apply(lemmatization)
                    input_vectors = tfidf.transform(df['lemmatized'])
                    df['Prediction'] = model.predict(input_vectors)
                    df['Sentiment'] = df['Prediction'].apply(lambda x: 'Positive' if x == 1 else 'Negative')
                    st.success("✅ Predictions complete!")
                    st.dataframe(df[['text', 'Sentiment']])

                    sentiment_counts = df['Sentiment'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index,
                           autopct='%1.1f%%', startangle=90,
                           colors=["#00cc66", "#ff6666"])
                    ax.axis('equal')
                    st.subheader("📊 Sentiment Distribution")
                    st.pyplot(fig)
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")

st.markdown("""
    <hr style="margin-top:40px;">
    <div style="text-align:center">
        <small>Made with using NLP(SpaCy), Streamlit, and Scikit-learn</small>
    </div>
""", unsafe_allow_html=True)
