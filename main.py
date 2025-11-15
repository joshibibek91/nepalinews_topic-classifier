import pandas as pd
import pickle

from fastapi import FastAPI

app = FastAPI()

dbfile = open('NepaliLogisticRegression.pickle', 'rb')
model = pickle.load(dbfile)


@app.get("/news_category/")
def read_item(news: str):
    news_data= {'predict_news':[news]}
    news_data_df= pd.DataFrame(news_data)
    
    df = pd.DataFrame({
        'news':[news],
    })
    
    result = model.predict(news_data_df['predict_news'])[0]
   
    # if int(result) == 0:
    #     Sentiment = "Negative" 
    # else:
    #     Sentiment= "Positive"
    
    return {"Category": result}
