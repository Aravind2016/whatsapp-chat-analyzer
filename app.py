import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
import preprocessor
import helper
import pandas as pd

st.sidebar.title("Whatsapp chat analyzer")
uploaded_file = st.sidebar.file_uploader("choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess_with_emotion(data)  # Use preprocess_with_emotion instead of preprocess
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "overall")
    selected_user = st.sidebar.selectbox("show Analysis wrt", user_list)
    if st.sidebar.button("show Analysis"):
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        if selected_user == 'overall':
            st.title('Most busy user')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        st.title('wordcloud')
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        st.title("Overall Sentiment Analysis")
        if selected_user != 'overall':
            df_selected_user = df[df['user'] == selected_user]
            user_sentiments = df_selected_user['message'].apply(helper.sentiment_analysis)
        else:
            user_sentiments = df['message'].apply(helper.sentiment_analysis)
        overall_sentiment = helper.overall_sentiment(user_sentiments)
        st.write(f"Overall Sentiment: {overall_sentiment}")

        st.title("Sentiment Analysis")
        sentiment_counts = user_sentiments.value_counts()

        # Display sentiment distribution
        sentiment_counts = user_sentiments.value_counts()
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        explode = (0.1, 0, 0)  # explode the 1st slice
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=None, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
        ax.set_title('Sentiment Distribution')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.legend(sentiment_counts.index, loc="best")  # Add legend with labels
        st.pyplot(fig)

        # Display emojis separately
        st.subheader("Sentiment Emojis")
        emoji_map = {
            "Positive": "😊",
            "Neutral": "😐",
            "Negative": "😞"
        }
        for sentiment, emoji in emoji_map.items():
            st.write(f"{sentiment}: {emoji}")

        # Display Emotion Analysis
        st.title("Emotion Analysis")
        if selected_user != 'overall':
            user_emotion = df[df['user'] == selected_user]['emotion'].mode().values[0]
            if selected_user!= 'group_notification':
                st.write(f"Emotion of {selected_user}: {user_emotion}")
        else:
            user_emotions = df.groupby('user')['emotion'].apply(lambda x: x.mode().values[0]).reset_index()
            user_emotions.columns = ['User', 'Emotion']  # Rename columns
            # Remove the last row from the dataframe
            user_emotions = user_emotions.iloc[:-1]
            # Display a table showing the emotions of each user
            st.write("Emotions of Each User:")
            st.dataframe(user_emotions)
