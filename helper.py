import pandas as pd
from collections import Counter
from textblob import TextBlob
from urlextract import URLExtract
from wordcloud import WordCloud

import pandas as pd

from textblob import TextBlob


from collections import Counter

extract=URLExtract()
def fetch_stats(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    words = []
    for message in df['message']:
        words.extend(message.split())
    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df, selected_user='overall'):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    x = df['user'].value_counts().head()
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df_percent

def create_wordcloud(selected_user, df, font_path=None):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def monthly_timeline(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def sentiment_analysis(message):
    analysis = TextBlob(message)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

def average_response_time(df, selected_user='overall'):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    df['message_date'] = pd.to_datetime(df['message_date'])
    df['response_time'] = df.groupby('user')['message_date'].diff()
    df['response_time_seconds'] = df['response_time'].dt.total_seconds()
    average_response_times = df.groupby('user')['response_time_seconds'].mean()
    return average_response_times
def overall_sentiment(sentiments):
    sentiment_counts = Counter(sentiments)
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return overall_sentiment

