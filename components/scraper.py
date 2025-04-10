import json
import os
import pandas as pd
import csv

from googleapiclient.discovery import build
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

credentialsPath = os.path.join(os.path.dirname(__file__), '..', 'credentials.json')
credentialsPath = os.path.abspath(credentialsPath)

# Load credentials
with open(credentialsPath, 'r') as f:
    credentials = json.load(f)

api_key = credentials["api_key"]
IDs_Video = credentials["video_id"]

trainDataPath = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'training.csv')
trainDataPath = os.path.abspath(trainDataPath)

# Load training data
trainData = pd.read_csv(trainDataPath)
xTrain = trainData["text"]
yTrain = trainData["label"]

# TF-IDF and Random Forest
vectorizer = TfidfVectorizer()
xTrainTF = vectorizer.fit_transform(xTrain)

clf = RandomForestClassifier()
clf.fit(xTrainTF, yTrain)

# Youtube API
youtube = build('youtube', 'v3', developerKey=api_key)

def getComments(video_id):
    comments = []
    nextPageToken = None
    try:
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,    
                maxResults=100,        
                pageToken=nextPageToken
            )

            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': comment['textDisplay'],
                })

            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                break
        print("✅ getComments Success")
        return comments
    except Exception as e:
        print(f"⚠️ Error to getComments {e}")
        return []

# Take Comment & Filter with Machine Learning
filterComments = []

for vid in IDs_Video:
    print(f"🔎 Search Comment from Video ID: {vid}")
    try:
        rawComments = getComments(vid)
        texts = [c['text'] for c in rawComments]
        if not texts:
            continue
        TfidfInput = vectorizer.transform(texts)
        predictions = clf.predict(TfidfInput)

        for i, label in enumerate(predictions):
            if label == 1:
                filterComments.append(texts[i]) 
        
        print("✅ Filtering Success")
    except Exception as e:
        print(f"❌ Search Comment from Video ID {vid} Failed: {e}")

if filterComments:
    df = pd.DataFrame(filterComments)
    outputPath = os.path.join(os.path.dirname(__file__), '..', 'filter', 'filter.csv')
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    df.to_csv(outputPath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    print(f"✅ ML Filtered Comment saved in {outputPath}")
else:
    print("⚠️ Commment not fullyfy the Criteria")