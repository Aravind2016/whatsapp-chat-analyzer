import re
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
nltk.data.path.append("C:\\Users\\aravi\\AppData\\Roaming\\nltk_data")


def detect_emotion(message):
    try:
        sid = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(message)
    return max(scores, key=scores.get)

# Function to preprocess data and add emotion column


def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Translate messages to English using translation service

    # Convert message_date type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df

def detect_emotion(message):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(message)
    if scores['compound'] >= 0.5:
        return 'Happy'
    elif 0.1 <= scores['compound'] < 0.5:
        return 'Neutral'
    elif -0.1 < scores['compound'] < 0.1:
        return 'Meh'
    elif -0.5 <= scores['compound'] < -0.1:
        return 'Sad'
    else:
        return 'Angry'

# Function to preprocess data with emotion detection
def preprocess_with_emotion(data):
    df = preprocess(data)
    df['emotion'] = df['message'].apply(detect_emotion)
    return df
