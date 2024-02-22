import os
from flask import Flask, request, jsonify
import dialogflow
from google.api_core.exceptions import InvalidArgument

app = Flask(__name__)

# Set your Dialogflow project details
DIALOGFLOW_PROJECT_ID = "chatanalyzerbot-rynq"
DIALOGFLOW_LANGUAGE_CODE = "en"
GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\aravi\Downloads\Telegram Desktop\agent.json"

# Create a Dialogflow session client
session_client = dialogflow.SessionsClient()

# Function to detect intent from user message using Dialogflow
def detect_intent(text_input, session_id):
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, session_id)
    text_input = dialogflow.types.TextInput(text=text_input, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)
    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
        return response.query_result.fulfillment_text
    except InvalidArgument:
        return "InvalidArgument error occurred"

# Define a route to handle incoming messages from users
@app.route("/webhook", methods=["POST"])
def webhook():
    request_data = request.get_json()
    user_message = request_data["message"]
    session_id = request_data["session_id"]
    bot_response = detect_intent(user_message, session_id)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
