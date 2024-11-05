import os
from dotenv import load_dotenv
import psycopg2


def insert_data(table_name, user_id, chat_id, message, sentiment_score,
                sentiment_level, insultiveness_status, chat_name, username,
                timestamp):
    """
    Inserts data into the specified table in the PostgreSQL database.

    Parameters:
    table_name (str): The name of the table where the data will be inserted.
    user_id (str): The ID of the user.
    chat_id (str): The ID of the chat.
    message (str): The message text.
    sentiment_score (float): The sentiment score of the message.
    timestamp (str): The timestamp of when the message was created (format: 'YYYY-MM-DD HH:MM:SS').
    """

    # Load environment variables from the .env file
    load_dotenv()

    # Establish the database connection
    connection = psycopg2.connect(database=os.getenv("DB_NAME"),
                                  host=os.getenv("DB_HOST"),
                                  user=os.getenv("DB_USER"),
                                  password=os.getenv("DB_PASSWORD"))

    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    # SQL query to insert data into the specified table
    insert_query = f"""
    INSERT INTO {table_name} (user_id, chat_id, message, sentiment_score,
                sentiment_level, insultiveness_status, chat_name, username, timestamp)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # The data you want to insert
    data = (user_id, chat_id, message, sentiment_score, sentiment_level,
            insultiveness_status, chat_name, username, timestamp)

    try:
        # Execute the query with the provided data
        cursor.execute(insert_query, data)

        # Commit the transaction to save the changes
        connection.commit()
        print(f"Data inserted successfully into {table_name}.")

    except Exception as e:
        # If there's an error, rollback the transaction
        connection.rollback()
        print(f"An error occurred: {e}")

    finally:
        # Close the cursor and connection to free resources
        cursor.close()
        connection.close()


def get_last_3_messages(chat_id, table_name):
    # Load environment variables from the .env file
    load_dotenv()

    # Establish the database connection
    connection = psycopg2.connect(database=os.getenv("DB_NAME"),
                                  host=os.getenv("DB_HOST"),
                                  user=os.getenv("DB_USER"),
                                  password=os.getenv("DB_PASSWORD"))

    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    try:
        # SQL query to fetch the last 10 messages for the given chat_id
        cursor.execute(
            f"""
            SELECT sentiment_score, sentiment_level, insultiveness_status, message,              username 
            FROM {table_name} 
            WHERE chat_id = %s 
            ORDER BY id DESC LIMIT 3;
        """, (chat_id, ))

        # Fetch the results
        last_3_messages = cursor.fetchall()
        return last_3_messages

    except Exception as e:
        # If there's an error, rollback the transaction
        connection.rollback()
        print(f"Error fetching last 3 messages: {e}")

    finally:
        # Close the cursor and connection to free resources
        cursor.close()
        connection.close()


def get_top_users_by_sentiment(chat_id, table_name):
    # Load environment variables from the .env file
    load_dotenv()

    # Establish the database connection
    connection = psycopg2.connect(database=os.getenv("DB_NAME"),
                                  host=os.getenv("DB_HOST"),
                                  user=os.getenv("DB_USER"),
                                  password=os.getenv("DB_PASSWORD"))

    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    try:
        # SQL query to fetch the top 5 users by sentiment score
        cursor.execute(
            f"""
            SELECT username, AVG(sentiment_score) as avg_sentiment
            FROM {table_name}
            WHERE chat_id = %s
            GROUP BY username
            ORDER BY avg_sentiment DESC
            LIMIT 5;
    """, (chat_id, ))

        # Fetch the results
        top_users = cursor.fetchall()
        return top_users

    except Exception as e:
        # If there's an error, rollback the transaction
        connection.rollback()
        print(f"Error fetching average sentiment: {e}")

    finally:
        # Close the cursor and connection to free resources
        cursor.close()
        connection.close()


# Example usage of the function
if __name__ == "__main__":
    None
