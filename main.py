import os
import time
import telebot
from dotenv import load_dotenv
from datetime import datetime, timezone
from alive import keep_alive
from openai import OpenAI
from psycopg import insert_data
from psycopg import get_last_3_messages
from psycopg import get_top_users_by_sentiment
import re
import matplotlib
import random

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt  # Correct position and formatting of import
import io

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials and API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BOT_TOKEN = os.getenv('BOT_TOKEN')
table_name = os.getenv('TABLE_NAME')

# Load the sentiment analysis prompt from a separate file
with open('sentiment_promt.txt', 'r', encoding='utf-8') as file:
    SENTIMENT_PROMPT = file.read()

# Initialize the OpenAI client and Telegram Bot using retrieved API keys
openai_client = OpenAI(api_key=OPENAI_API_KEY)
bot = telebot.TeleBot(BOT_TOKEN)

# Keep the service alive to avoid timeouts
keep_alive()

# Global variable to store the current admin for notifications
current_admin = None

# Function to clean the text by removing unwanted elements like links, numbers, emojis, and special characters
def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove website links
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove emojis and special characters
    text = text.strip()  # Remove any extra spaces
    return text


# Function to retrieve the usernames of all administrators in a given chat
def get_admin_usernames(chat_id):
    admins = bot.get_chat_administrators(chat_id)
    admin_usernames = [
        admin.user.username for admin in admins if admin.user.username
    ]
    return admin_usernames

# Function to check if the user is an admin
def is_admin(message):
    """
    Checks if the user sending a message is an administrator in the given chat.
    
    Args:
        message (telebot.types.Message): The message object containing information about the sender.
    
    Returns:
        bool: True if the user is an admin, False otherwise.
    """
    admin_usernames = get_admin_usernames(message.chat.id)
    return message.from_user.username in admin_usernames

# Command handler for /last3
@bot.message_handler(commands=['last3'])
def handle_last3_command(message):
    if not is_admin(message):
        bot.reply_to(message, "Эта команда доступна только администраторам.")
        return
    
    chat_id = str(message.chat.id)

    # Fetch the last 10 messages from the database for the specific chat
    last_3_messages = get_last_3_messages(chat_id, table_name)

    # Format the message to display the last 10 sentiments
    if last_3_messages:
        response_message = "Last 3 messages' sentiments:\n\n"
        for idx, record in enumerate(last_3_messages, 1):
            response_message += f"{idx}. Username: {record[4]}. \n Text: {record[3]} \n Sentiment Score: {record[0]}\n Sentiment Level: {record[1]}\n Insultiveness: {record[2]}\n\n"
    else:
        response_message = "No messages found in this chat."

    # Send the response to the user
    bot.send_message(chat_id, response_message)


# Command handler for /barchart
@bot.message_handler(commands=['barchart'])
def handle_barchart_command(message):
    if not is_admin(message):
        bot.reply_to(message, "Эта команда доступна только администраторам.")
        return
    
    chat_id = str(message.chat.id)
    print(f"Handling /barchart for chat_id: {chat_id}")  # Debug log

    top_users = get_top_users_by_sentiment(chat_id, table_name)
    print(f"Top users data: {top_users}")  # Debug log

    if top_users:
        # Filter out None values from usernames and corresponding sentiment levels
        filtered_users = [(user, sentiment) for user, sentiment in top_users
                          if user is not None]

        usernames = [user for user, sentiment in filtered_users]
        sentiment_levels = [
            float(sentiment) for user, sentiment in filtered_users
        ]

        if usernames and sentiment_levels:
            # Create bar chart
            plt.figure(figsize=(10, 6))
            plt.barh(usernames, sentiment_levels, color='skyblue')
            plt.xlabel('Average Sentiment Level')
            plt.title('Top 5 Users by Average Sentiment Level')

            # Save chart to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            # Send chart as an image
            bot.send_photo(chat_id, photo=buf)
        else:
            bot.send_message(chat_id,
                             "No valid data available to generate chart.")
    else:
        bot.send_message(chat_id, "No data available to generate chart.")

# Command handler for /update_admin
@bot.message_handler(commands=['update_admin'])
def handle_update_admin(message):
    """
    Update the current admin for notification purposes.

    This command is available only to administrators.
    The format of the command is `/update_admin <new_admin_username>`.
    If no username is provided, the bot will reply with a usage message.
    """
    if not is_admin(message):
        bot.reply_to(message, "Эта команда доступна только администраторам.")
        return

    global current_admin
    if len(message.text.split()) > 1:
        new_admin = message.text.split()[1]
        current_admin = new_admin
        bot.reply_to(message, f"Текущий админ для уведомлений обновлен: @{new_admin}")
    else:
        bot.reply_to(message, "Пожалуйста, укажите username нового администратора.")

# Function to handle regular text messages
@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_text(message):
    # Check if the message is not a command
    if message.text.startswith('/'):
        return

    # Extract relevant information from the message
    chat_id = str(message.chat.id)
    chat_name = str(message.chat.title)
    text = clean_text(message.text.strip().lower())
    user_id = str(message.from_user.id)
    username = str(message.from_user.username)
    unix_timestamp = message.date
    utc_time = datetime.fromtimestamp(
        unix_timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # Get sentiment analysis from OpenAI's GPT model
    sentiment_score = round(float(chat_with_gpt(text).split(",")[0]), 2)
    sentiment_level = chat_with_gpt(text).split(",")[1].strip()
    insultiveness_status = chat_with_gpt(text).split(",")[2].strip()

    # Insert the analyzed data into the database
    insert_data(table_name, user_id, chat_id, text, sentiment_score,
                sentiment_level, insultiveness_status, chat_name, username,
                utc_time)

    # If the message is highly negative and insultive, notify the admin
    if (sentiment_score
            < -0.8) and (sentiment_level
                         == 'Highly Negative') and (insultiveness_status
                                                    == 'Insultive'):
        notify_admin(chat_id, username, message.message_id)


# Function to interact with OpenAI's GPT model to get sentiment analysis
def chat_with_gpt(text):
    completion = openai_client.chat.completions.create(model="gpt-4o-mini",
                                                       messages=[{
                                                           "role":
                                                           "system",
                                                           "content":
                                                           SENTIMENT_PROMPT
                                                       }, {
                                                           "role":
                                                           "user",
                                                           "content":
                                                           text
                                                       }],
                                                       temperature=0.6)
    return completion.choices[0].message.content.strip()


# Function to notify the admin in the chat if the sentiment score is below -0.9
def notify_admin(chat_id, user_id, message_id):
    """
    Notify the admin in the chat if the sentiment score is below -0.9

    The function will first check if there is a current admin assigned. If there is,
    it will send a message to the admin with the user's message ID and the sentiment
    score. If there is no current admin, it will choose a random admin from the list
    of admins in the chat and send the message to them. If there are no admins in the
    chat, it will send a message to the chat saying that the user's message was highly
    negative but there are no admins to notify.
    """
    global current_admin
    admins = get_admin_usernames(chat_id)  # Получаем список администраторов

    if current_admin:  # Если текущий админ назначен
        bot.reply_to(
            message_id,
            f"Внимание, пользователь @{user_id} отправил крайне \
                негативное сообщение!\n\n@{current_admin}, обратите внимание"
        )
    elif admins:  # Если текущий админ не назначен, выбираем случайного
        random_admin = random.choice(admins)
        bot.reply_to(
            message_id, 
            f"Внимание, пользователь @{user_id} отправил крайне \
                негативное сообщение!\n\n@{random_admin}, обратите внимание"
        )
    else:  # Если админов нет
        bot.reply_to(message_id, f"Внимание, пользователь @{user_id} отправил крайне \
            негативное сообщение, но администраторы не найдены.")



# Function to start the bot with exception handling to prevent crashes
def run_bot():
    while True:
        try:
            bot.polling()
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(5)  # Wait for a few seconds before restarting
            continue  # Retry polling


# Main execution block to start the bot
if __name__ == "__main__":
    run_bot()
