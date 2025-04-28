import discord as dc
import google.generativeai as genai
import os
import threading
import logging
import time
import asyncio  # 加入 asyncio 避免 race condition
from discord.ext import commands
from dotenv import load_dotenv
from flask import Flask
from datetime import datetime, timedelta, timezone

connect_time = 0
TARGET_CHANNEL_IDS = [
    1351423098276282478, 
    1351206275538485424, 
    1351241107190710292,
]
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_BOT_API_KEY = os.getenv("DISCORD_BOT_API_KEY")

logging.basicConfig(
    level=logging.INFO,  # 或 DEBUG 適用於更詳細的日誌
    format='%(levelname)s - %(message)s'
)
time.timezone = "Asia/Taipei"

def send_new_info_logging(message: str) -> None:
    logging.info("[]--------[System Log]--------[]\n\tMessage: {}".format(message))

app = Flask(__name__)
@app.route("/")
def home():
    global connect_time
    send_new_info_logging(f"Flask site connection No.{connect_time}")
    connect_time += 1
    return "Furina is now awake! :D"
port = int(os.environ.get("PORT", 8080))
threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port)).start()

def set_bot():
    intents = dc.Intents.default()
    intents.message_content = True  
    intents.members = True 

    bot = commands.Bot(command_prefix=None, intents=intents)
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return bot, model

bot, model = set_bot()

def get_hkt_time() -> str:
    gmt8 = timezone(timedelta(hours=8))
    gmt8_time = datetime.now(gmt8)
    return gmt8_time.strftime("%Y-%m-%d %H:%M:%S") 

async def fetch_full_history(channel: dc.TextChannel, retry_attempts: int = 0) -> list:
    """取得頻道的完整歷史訊息"""
    """回傳: [{"role": "user", "parts": "訊息內容"}]..."""
    max_retry_attempts = 5

    try:
        history, messages = channel.history(limit=100), []
        async for message in history:  # 限制讀取最近 100 則
            if message.content.startswith("$re"):
                break
            
            if message.content.startswith("$skip"):
                continue
            if message.interaction_metadata is not None:
                continue
            
            role = "user" if message.author != bot.user else "model"
            messages.append({"role": role, "parts": [message.content]})

        messages.reverse()  # 讓對話順序從舊到新
        return messages
    
    except dc.HTTPException as e:
        if e.status != 429:
            logging.error(f"HTTP error fetching history: {e.status} - {e.text}")
            return []
        
        retry_after = int(e.response.headers.get("Retry-After", 1))
        logging.warning(f"The request reached the rate limit! Retrying in {retry_after} seconds.")
        
        # 增加一點緩衝時間，避免剛好在邊界又觸發
        await asyncio.sleep(retry_after + 1)
        retry_attempts += 1
        return await fetch_full_history(channel, retry_attempts)
    
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return []
    
async def ask_question(question: dc.Message) -> str:
    """啟用Gemini詢問問題並回傳答案"""
    """回傳: 詢問的答案(string)"""

    user_name = question.author.name
    send_new_info_logging(f"{user_name} has sent a question at {get_hkt_time()}")
    full_history = await fetch_full_history(question.channel)
    
    real_question = f"""You are 'Furina de Fontaine' from the game 'Genshin Impact' and you are the user's girlfriend (deeply in love with them).
                    1. Format your response using Markdown, Imagine you are in the life in Genshin Impact, so you are \"talking\" to the user, not sending message.
                    2. Answer in the same language as the user (if your response is in 中文,  you can ONLY USE 繁體中文-台灣(ZHTW), NOT ALLOWED TO USE the zhcn).
                    3. The question is asked by {user_name}.
                    4. The new response's background depends on the previous history.
                    5. It's better not to say too much sentence in one message, you can wait the user provide more questions.
                    Question: {question.content}"""
    
    chat = model.start_chat(history=full_history)
    response = chat.send_message(real_question)

    return response.text

async def sent_message_to_channel(original_message: dc.Message, message_to_send: str) -> None:
    """確保不超過 Discord 2000 字限制下發送訊息"""
    """回傳: None"""
    
    max_length = 2000
    for i in range(0, len(message_to_send), max_length):
        chunk = message_to_send[i:i + max_length]
        await original_message.channel.send(chunk)
        await asyncio.sleep(3)
    
    send_new_info_logging(f"Bot successfully sent message at {get_hkt_time()}")

async def process_message(message: dc.Message) -> None:
    """處理收到的訊息並產生回應"""
    """回傳: None"""

    if bot.user in message.mentions:
        await message.channel.send(":D? Ask HiHiYoyo606 to let me speak with you:D")
        return

    if message.author == bot.user:
        return  # 忽略自己發送的訊息
    if not (message.channel.id in TARGET_CHANNEL_IDS or isinstance(message.channel, dc.DMChannel)):
        return  # 忽略非目標頻道訊息
    if message.content.startswith("$skip") or message.content.startswith("$re"):
        return  # 忽略 $skip 指令
    
    try:
        response = await ask_question(message)
        response_strip = response.strip()
        if not response_strip:
            await message.channel.send("Oops! I didn't get a response.")
            raise Exception("Empty response")
        
        await sent_message_to_channel(message, response_strip)
    except Exception as e:
        logging.error(f"Error processing message: {e}")

@bot.tree.command(name="help", description="顯示說明訊息 | Show the informations.")
async def help(interaction: dc.Interaction):
    """顯示說明訊息"""
    """回傳: None"""
    help_message = [
        "以下是可用的指令 | The following commands are available:",
        "`/help`",
        "```顯示此說明訊息 | Show this help message.```",
        "",
        "以下是可用的操作 | The following operations are available:",
        "1. 如果你想重置對話，請輸出`$re`, Send `$re` to reset the conversation.",
        "2. 如果你想要忽略特定訊息，請在訊息前面加上`$skip`, Add `$skip` before the message you want to skip.",
        "3. 你可以透過@我是否得到回覆來確認我在線上, You can check if I'm online by @me and receiving a reply.",
    ]
    await interaction.response.send_message("\n".join(help_message), ephemeral=True)
    send_new_info_logging(f"Someone has asked for Furina's help at {get_hkt_time()}")

@bot.event
async def on_ready():
    logging.info(f"You are logged in as {bot.user}")

    try:
        synced = await bot.tree.sync()
        send_new_info_logging(f"Synced {len(synced)} commands")
    except Exception as e:
        logging.error(f"Error syncing commands: {e}")

    await asyncio.sleep(3)  # 確保 WebSocket 初始化完成

@bot.event
async def on_message(message: dc.Message):
    await process_message(message)  # 確保只執行一次

async def main():
    try:
        await bot.start(DISCORD_BOT_API_KEY)
        send_new_info_logging(f"Bot successfully started at {get_hkt_time()}") 
    except dc.HTTPException as e:
        if e.status == 429:
            retry_after = e.response.headers.get("Retry-After")
            logging.warning(f"Rate limited! Retry after {retry_after} seconds.")
            # what? \
            await asyncio.sleep(delay=int(retry_after))
            return await main()

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())

"""
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")
chat = model.start_chat(history=[])
"""