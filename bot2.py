import discord
import google.generativeai as genai
import os
import logging
import asyncio
import threading
from flask import Flask
from dotenv import load_dotenv
from sklearn_model import MainFunctions

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
print(DISCORD_BOT_TOKEN)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gemini Model Setup ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash') # Or your preferred Gemini model
    logging.info("Gemini model initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Gemini model: {e}")
    model = None

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
bot = discord.Client(intents=intents)

# flask server
app = Flask(__name__)
@app.route("/")
def home():
    return "200 OK"
threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()

def get_message_url(guild_id: int, channel_id: int, message_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

@bot.event
async def on_ready():
    logging.info(f'Bot logged in as {bot.user.name}')
    logging.info(f'Bot ID: {bot.user.id}')
    print(f'Logged in as {bot.user.name}') # For quick console confirmation

async def send_long_message(
    destination: discord.abc.Messageable, 
    text: str, 
    max_length: int = 2000, 
    chunk_delay_seconds: float = 0.5
) -> None:
    """
    Sends a long message to a Discord destination, splitting it into chunks
    if it exceeds the specified maximum length (defaulting to Discord's 2000 character limit).

    Args:
        destination: The discord.abc.Messageable object (e.g., discord.TextChannel,
                     discord.User, or message.channel) to send the message to.
        text: The string content of the message to send.
        max_length: The maximum character length for each message chunk.
        chunk_delay_seconds: Delay in seconds between sending message chunks.
                             This can help avoid rate limits. Set to 0 for no delay.
    """
    if not text:  # Do nothing if the text is empty
        return
    current_pos = 0
    while current_pos < len(text):
        chunk = text[current_pos : current_pos + max_length]
        current_pos += len(chunk) # Ensure current_pos advances by the actual chunk length
        
        await destination.send(chunk)
        
        # Only sleep if there are more chunks to send and a delay is specified
        if current_pos < len(text) and chunk_delay_seconds > 0:
            await asyncio.sleep(chunk_delay_seconds)


@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if the bot is mentioned
    if bot.user.mentioned_in(message):
        if not model:
            await message.channel.send("Sorry, the Gemini model is not available at the moment.")
            return

        query = message.content
        for mention in message.mentions:
            if mention == bot.user:
                query = query.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '').strip()
                break
        
        if not query:
            await message.channel.send("Hello! How can I help you today?")
            return
        
        if query.endswith("$gtd"):
            await message.channel.send("```\n" + MainFunctions.get_training_data() + "\n```")
            return

        logging.info(f"Received query from {message.author.name}: \"{query}\"")
        
        try:
            """
            async with message.channel.typing():
                response = model.generate_content(query)
                if response.text:
                    # Use the new function here
                    await send_long_message(message.channel, response.text, chunk_delay_seconds=0.5)
                    logging.info(f"Sent Gemini response to {message.author.name}")
                else:
                    await message.channel.send("I received an empty response from the model.")
                    logging.warning("Gemini model returned an empty response.")
                """
            response = await MainFunctions.get_label(query)
            result = response.get("結果 Result", None)
            if not result:
                raise Exception("No result in response")
            
            if result == "普通 Normal":
                return

            message_url = get_message_url(message.guild.id, message.channel.id, message.id)
            await send_long_message(message.channel, f"{message_url} 疑似詐騙訊息，請注意。", chunk_delay_seconds=0.5)
        except Exception as e:
            logging.error(f"Error generating response from Gemini: {e}")
            await message.channel.send("Sorry, I encountered an error trying to respond.")

def run_bot():
    if not DISCORD_BOT_TOKEN:
        logging.error("DISCORD_BOT_TOKEN not found in environment variables.")
        return
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not found in environment variables.")
        # You might still want to run the bot without Gemini for basic functionality
        # or handle this more gracefully depending on requirements.
    
    try:
        bot.run(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logging.error("Failed to log in. Please check your Discord Bot Token.")
    except Exception as e:
        logging.error(f"An error occurred while running the bot: {e}")

if __name__ == "__main__":
    run_bot()
