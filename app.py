
import streamlit as st
import pandas as pd
import pickle


st.title("📰 Nepali News Topic Classifier")
# -----------------------------
# 🧩 Load the trained model
# -----------------------------
dbfile = open('NepaliLogisticRegression.pickle', 'rb')
model = pickle.load(dbfile)


st.write("यहाँ कुनै पनि नेपाली समाचार टाइप गर्नुहोस्, म यसले कुन विषयसँग सम्बन्धित छ भनेर अनुमान गर्छु।")

# -----------------------------
# 🧹 Preprocessing Function
# -----------------------------

# -----------------------------
# 📝 User Input
# -----------------------------
text = st.text_area("कृपया समाचार लेख्नुहोस्:", placeholder="उदाहरण: प्रधानमन्त्रीले नयाँ नीति घोषणा गरे")
news_data = {'predict_news':[text]}
news_data_df = pd.DataFrame(news_data)

if st.button("🔮 विषय अनुमान गर्नुहोस्"):
    if not text.strip():
        st.warning("कृपया समाचार लेख्नुहोस्।")
    else:
        df = pd.DataFrame({
               'news': [text],
                })
        # st.dataframe(df)
        result = model.predict(news_data_df['predict_news'])

        # Show cleaned text and result
        st.subheader("समाचार:")
        st.write(news_data_df['predict_news'][0])

        st.subheader("अनुमान गरिएको विषय:")
        st.success(result[0])

    
