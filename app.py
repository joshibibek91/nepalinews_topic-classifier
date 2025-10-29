
import streamlit as st
import pandas as pd
import pickle
import re

# -----------------------------
# 🧩 Load the trained model
# -----------------------------
with open('NepaliLogisticRegression.pickle', 'rb') as f:
    model = pickle.load(f)

st.title("📰 Nepali News Topic Classifier")
st.write("यहाँ कुनै पनि नेपाली समाचार टाइप गर्नुहोस्, म यसले कुन विषयसँग सम्बन्धित छ भनेर अनुमान गर्छु।")

# -----------------------------
# 🧹 Preprocessing Function
# -----------------------------
# Make sure this matches the same function used during training
stopwords = [
    'छ', 'छन्', 'थिए', 'गरे', 'गर्नु', 'गरेको', 'गर्छ', 'गर्न', 'पनि',
    'भएको', 'भए', 'हो', 'र', 'लाई', 'मा', 'बाट', 'संग', 'कि', 'तथा'
]
suffixes = ['हरु', 'हरू', 'को', 'ले', 'मा', 'बाट', 'का', 'की', 'हो', 'गरेको', 'गरे', 'गर्ने', 'गर्छ']

def clean_and_stem(text):
    text = str(text)
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    words = text.split()
    # stem suffixes
    for i in range(len(words)):
        for suf in suffixes:
            if words[i].endswith(suf) and len(words[i]) > len(suf) + 1:
                words[i] = words[i][:-len(suf)]
                break
    words = [w for w in words if w not in stopwords]
    return " ".join(words)

# -----------------------------
# 📝 User Input
# -----------------------------
text = st.text_area("कृपया समाचार लेख्नुहोस्:", placeholder="उदाहरण: प्रधानमन्त्रीले नयाँ नीति घोषणा गरे")

if st.button("🔮 विषय अनुमान गर्नुहोस्"):
    if not text.strip():
        st.warning("कृपया समाचार लेख्नुहोस्।")
    else:
        cleaned_text = clean_and_stem(text)
        prediction = model.predict([cleaned_text])[0]

        # Show cleaned text and result
        st.subheader("समाचार:")
        st.write(cleaned_text)

        st.subheader("अनुमान गरिएको विषय:")
        st.success(prediction)

        st.balloons()
