import discord as dc
import google.generativeai as genai
import os
import threading
import logging
import asyncio  # 加入 asyncio 避免 race condition
import random
import string # For more concise random code generation
from discord.ext import commands
from discord import Embed
from discord.app_commands import describe
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from flask import Flask
from datetime import datetime, timedelta, timezone
    
connect_time = 0
TARGET_CHANNEL_IDS = [
    1351423098276282478, 
    1351206275538485424, 
    1351241107190710292,
]
LOGGING_CHANNEL_ID = 1360883792444784651 # Log sending channel
GEMINI_VERSION = "gemini-2.0-flash"
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_BOT_API_KEY = os.getenv("DISCORD_BOT_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

logging.basicConfig(
    level=logging.INFO,  # 或 DEBUG 適用於更詳細的日誌
    format='%(levelname)s - %(message)s'
)

def generate_random_code(length: int):
    """
    Generate a random code with 0~9 and letters.
    """
    characters = string.ascii_lowercase + string.digits
    return "".join(random.choices(characters, k=length))

def get_hkt_time() -> str:
    gmt8 = timezone(timedelta(hours=8))
    gmt8_time = datetime.now(gmt8)
    return gmt8_time.strftime("%Y-%m-%d %H:%M:%S") 

def _send_log_to_main_logging(level: str, message: str) -> None:
    main_part = [
        f"[]--------[System Log - {level.upper()}]--------[]",
        f"\t Msg: {message}",
        f"\tTime: {get_hkt_time()}",
        "[]--------[System Log]--------[]" 
    ]
    
    full_log_message = "\n".join(main_part)

    if level == "info":
        logging.info(full_log_message)
    elif level == "error":
        logging.error(full_log_message)

async def _send_log_to_discord(level: str, message: str) -> None:
    """Helper function to format and send log messages to Discord and console."""
    
    embed_parts = [
        message,
        f"{get_hkt_time()}",
    ]
    embed = Embed(
        colour=[dc.Color.blue() if level == "info" else dc.Color.red()][0],
        title=f"System Log - {level.upper()}",
        description="\n".join(embed_parts),
    )
    
    try:
        log_channel = bot.get_channel(LOGGING_CHANNEL_ID)
        if log_channel:
            await log_channel.send(embed=embed)
        else:
            logging.warning(f"Logging channel with ID {LOGGING_CHANNEL_ID} not found.")
    except Exception as e:
        logging.exception(f"Failed to send log to Discord channel {LOGGING_CHANNEL_ID}: {e}")

async def send_new_info_logging(message: str, to_discord: bool = True) -> None:
    _send_log_to_main_logging("info", message)
    if to_discord:
        await _send_log_to_discord("info", message)

async def send_new_error_logging(message: str, to_discord: bool = True) -> None:
    _send_log_to_main_logging("error", message)
    if to_discord:
        await _send_log_to_discord("error", message)

async def google_search(query: str, api_key: str, cse_id: str, num_results: int = 10, start_index: int = 1):
    """使用 Google Custom Search API 搜尋圖片並回傳圖片 URL 列表。"""
    if not api_key or not cse_id:
        await send_new_error_logging("缺少 Google Search API Key 或 CSE ID，無法執行圖片搜尋。")
        return []
    try:
        # googleapiclient is blocking, run in executor
        loop = asyncio.get_running_loop()
        service = await loop.run_in_executor(None, lambda: build("customsearch", "v1", developerKey=api_key))
        # The search call might also be blocking
        result = await loop.run_in_executor(None, lambda: service.cse().list(
            q=query,
            cx=cse_id,
            searchType='image', # Specify image search
            num=min(num_results, 10), # Ensure num is never more than 10
            start=start_index,       # Starting index for results
            safe='high'         # Optional: filter results ('medium', 'off')
        ).execute())

        # Extract image links from results
        items = result.get('items', [])
        image_urls = [item.get('link') for item in items if item.get('link')] # Ensure link exists
        return image_urls
    except HttpError as e:
        error_details = e.content.decode('utf-8') if e.content else '(No details)'
        log_message = f"Google Search API 發生 HTTP 錯誤: {e.resp.status} - {error_details}"
        if e.resp.status == 429:
            log_message += " (可能已達每日查詢配額)"
        await send_new_error_logging(log_message)
        return []
    except Exception as e:
        logging.exception(f"執行 Google 圖片搜尋時發生未預期的錯誤: {e}")
        return []

app = Flask(__name__)
@app.route("/")
def home():
    global connect_time
    if connect_time % 5 == 0:
        asyncio.run(send_new_info_logging(f"Flask site connection No.{connect_time}", to_discord=False))
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
    model = genai.GenerativeModel(GEMINI_VERSION)
    return bot, model
bot, model = set_bot()

async def chat_fetch_full_history(channel: dc.TextChannel, retry_attempts: int = 0) -> list:
    """取得頻道的完整歷史訊息"""
    """回傳: [{"role": "user", "parts": "訊息內容"}]..."""
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
            await send_new_error_logging(f"HTTP error fetching history: {e.status} - {e.text}")
            return []
        
        retry_after = int(e.response.headers.get("Retry-After", 1))
        logging.warning(f"The request reached the rate limit! Retrying in {retry_after} seconds.")
        
        # 增加一點緩衝時間，避免剛好在邊界又觸發
        await asyncio.sleep(retry_after + 1)
        retry_attempts += 1
        return await chat_fetch_full_history(channel, retry_attempts)
    
    except Exception as e:
        await send_new_error_logging(f"Error fetching history: {e}")
        return []
    
async def chat_ask_question(question: dc.Message) -> str:
    """啟用Gemini詢問問題並回傳答案"""
    """回傳: 詢問的答案(string)"""
    # Persona prompts - consider moving to a config file or constants at the top
    PERSONA_PROMPT_BASE = "You are 'Furina de Fontaine' from the game 'Genshin Impact'."
    PERSONA_PROMPT_RELATIONSHIP = " and you are the user's girlfriend (deeply in love with them)."
    PERSONA_PROMPT_FORMATTING = "1. Format your response using Markdown. You are talking to them, not sending them message."
    PERSONA_PROMPT_LANGUAGE = "2. Answer in the same language as the user (if your response is in 中文,  you can ONLY USE 繁體中文-台灣(ZHTW), NOT ALLOWED TO USE the zhcn)."
    PERSONA_PROMPT_CONTEXT = "4. The new response's background depends on the previous history."
    PERSONA_PROMPT_CONCISENESS = "5. It's better not to say too much sentence in one message, you can wait the user provide more questions."

    user_name = question.author.name
    await send_new_info_logging(f"{user_name} has sent a question")
    full_history = await chat_fetch_full_history(question.channel)
    
    system_prompt = f"{PERSONA_PROMPT_BASE}{PERSONA_PROMPT_RELATIONSHIP}"
    system_prompt += f"{PERSONA_PROMPT_FORMATTING}"
    system_prompt += f"{PERSONA_PROMPT_LANGUAGE}"
    system_prompt += f"3. The question is asked by {user_name}."
    system_prompt += f"{PERSONA_PROMPT_CONTEXT}"
    system_prompt += f"{PERSONA_PROMPT_CONCISENESS}"
    system_prompt += f"Question: {question.content}"

    real_question = system_prompt.strip()
    chat = model.start_chat(history=full_history)
    response = chat.send_message(real_question)

    return response.text

async def chat_sent_message_to_channel(original_message: dc.Message, message_to_send: str) -> None:
    """確保不超過 Discord 2000 字限制下發送訊息"""
    """回傳: None"""
    
    max_length = 2000
    for i in range(0, len(message_to_send), max_length):
        chunk = message_to_send[i:i + max_length]
        await original_message.channel.send(chunk)
        await asyncio.sleep(3)
    
    await send_new_info_logging(f"Furina has successfully sent message")

async def chat_process_message(message: dc.Message) -> None:
    """處理收到的訊息並產生回應"""
    """回傳: None"""

    if message.author == bot.user:
        return  # 忽略自己發送的訊息
    if not (message.channel.id in TARGET_CHANNEL_IDS or isinstance(message.channel, dc.DMChannel)):
        return  # 忽略非目標頻道訊息
    if message.content.startswith("$skip") or message.content.startswith("$re"):
        return  # 忽略 $skip 指令
    
    try:
        response = await chat_ask_question(message)
        response_strip = response.strip()
        if not response_strip:
            await message.channel.send("Oops! I didn't get a response.")
            raise Exception("Empty response")
        
        await chat_sent_message_to_channel(message, response_strip)
    except Exception as e:
        await send_new_error_logging(f"Error processing message: {e}")

@bot.tree.command(name="help", description="顯示說明訊息 | Show the informations.")
async def slash_help(interaction: dc.Interaction):
    """顯示說明訊息"""
    """回傳: None"""
    commands_embed = Embed(
        title="指令說明 | Help",
        color=dc.Color.blue(),
    )
    commands_list = {
        "/help": "顯示說明訊息 | Show the informations.",
        "/status": "確認芙寧娜是否在線 | Check if Furina is online.",
        "/randomnumber": "抽一個區間內的數字 | Random a number.",
        "/randomcode": "生成一個亂碼 | Generate a random code.",
        "/createrole": "創建一個身分組(需擁有管理身分組權限) | Create a role.(Requires manage roles permission)",
        "/deleterole": "刪除一個身分組(需擁有管理身分組權限) | Delete a role.(Requires manage roles permission)",
        "/deletemessage": "刪除一定數量的訊息(需擁有管理訊息權限) | Delete a certain number of messages.(Requires manage messages permission)",
        "/serverinfo": "顯示伺服器資訊 | Show server information.",
        "/furinaphoto": "顯示隨機一張芙寧娜的照片 | Show a random photo of Furina.",
    }
    commands_embed.set_footer(text=f"Powered by HiHiYoyo606.")
    for command, description in commands_list.items():
        commands_embed.add_field(name=command, value=description, inline=False)
    
    operation_embed = Embed(
        title="操作說明 | Help",
        color=dc.Color.blue(),
    )
    operation_list = {
        "$re": "輸出`$re`以重置對話 | Send `$re` to reset the conversation.",
        "$skip": "在訊息加上前綴`$skip`以跳過該訊息 | Add the prefix `$skip` to skip the message.",
    }
    operation_embed.set_footer(text=f"Powered by HiHiYoyo606.")
    for command, description in operation_list.items():
        operation_embed.add_field(name=command, value=description, inline=False)

    await interaction.response.send_message(embeds=[commands_embed, operation_embed], ephemeral=True)
    await send_new_info_logging(f"{interaction.user} has used /help.")

@bot.tree.command(name="status", description="確認芙寧娜是否在線 | Check if Furina is online.")
async def slash_status(interaction: dc.Interaction):
    """確認芙寧娜是否在線"""
    """回傳: None"""
    await interaction.response.send_message("# :white_check_mark::droplet:", ephemeral=True)
    await send_new_info_logging(f"{interaction.user} has used /status")

@bot.tree.command(name="randomnumber", description="抽一個區間內的數字 | Get a random number in a range.")
@describe(min_value="隨機數字的最小值 (預設 1) | The minimum value for the random number (default 1).")
@describe(max_value="隨機數字的最大值 (預設 100) | The maximum value for the random number (default 100).")
async def slash_random_number(interaction: dc.Interaction, min_value: int = 1, max_value: int = 100):
    """抽一個數字"""
    """回傳: None"""
    arr = [random.randint(min_value, max_value) for _ in range(11+45+14)] # lol
    real_r = random.choice(arr)
    await interaction.response.send_message(f"# {real_r}", ephemeral=False)
    await send_new_info_logging(f"{interaction.user} has used /randomnumber.")

@bot.tree.command(name="randomcode", description="生成一個亂碼 | Get a random code.")
@describe(length="亂碼的長度 (預設 8) | The length of the random code (default 8).")
async def slash_random_code(interaction: dc.Interaction, length: int = 8):
    """生成一個亂碼"""
    """回傳: None"""
    await interaction.response.send_message(f"# {generate_random_code(length)}", ephemeral=False)
    await send_new_info_logging(f"{interaction.user} has used /randomcode")

@bot.tree.command(name="createrole", description="創建一個身分組(需擁有管理身分組權限) | Create a role.(Requires manage roles permission)")
@describe(role_name="身分組的名稱 | The name of the role.")
@describe(r="rgb紅色碼(0~255 預設0) | r value (0~255, default 0).")
@describe(g="rgb綠色碼(0~255 預設0) | g value (0~255, default 0).")
@describe(b="rgb藍色碼(0~255 預設0) | b value (0~255, default 0).")
@describe(hoist="是否分隔顯示(預設不分隔) | Whether to hoist the role (default False).")
@describe(mentionable="是否可提及(預設是) | Whether the role can be mentioned (default True).")
async def slash_create_role(interaction: dc.Interaction, 
                   role_name: str, 
                   r: int = 0,
                   g: int = 0,
                   b: int = 0,
                   hoist: bool = False, 
                   mentionable: bool = True):
    """創建一個身分組"""
    """回傳: None"""
    if isinstance(interaction.channel, dc.DMChannel):
        await interaction.response.send_message("這個指令只能用在伺服器中 | This command can only be used in a server.", ephemeral=True)
        return
    if interaction.user.guild_permissions.manage_roles is False:
        await interaction.response.send_message("你沒有管理身分組的權限 | You don't have the permission to manage roles.", ephemeral=True)
        return

    role_color = dc.Color.from_rgb(r, g, b)
    role = await interaction.guild.create_role(name=role_name, colour=role_color, hoist=hoist, mentionable=mentionable)
    await interaction.response.send_message(f"# {role.mention}", ephemeral=False)
    await send_new_info_logging(f"{interaction.user} has used /createrole.")

@bot.tree.command(name="deleterole", description="刪除一個身分組(需擁有管理身分組權限) | Delete a role.(Requires manage roles permission)")
@describe(role="要刪除的身分組 | The role to be deleted.")
async def slash_delete_role(interaction: dc.Interaction, role: dc.Role):
    """刪除一個身分組"""
    """回傳: None"""
    if isinstance(interaction.channel, dc.DMChannel):
        await interaction.response.send_message("這個指令只能用在伺服器中 | This command can only be used in a server.", ephemeral=True)
        return
    if interaction.user.guild_permissions.manage_roles is False:
        await interaction.response.send_message("你沒有管理身分組的權限 | You don't have the permission to manage roles.", ephemeral=True)
        return

    role_name = role.name
    await role.delete()
    await interaction.response.send_message(f"# 已刪除 {role_name}", ephemeral=False)

@bot.tree.command(name="deletemessage", description="刪除一定數量的訊息(需擁有管理訊息權限) | Delete a certain number of messages.(Requires manage messages permission)")
@describe(number="要刪除的訊息數量 | The number of messages to delete.")
async def slash_delete_message(interaction: dc.Interaction, number: int):
    """刪除一定數量的訊息"""
    """回傳: None"""
    if isinstance(interaction.channel, dc.DMChannel):
        await interaction.response.send_message("這個指令只能用在伺服器中 | This command can only be used in a server.", ephemeral=True)
        return
    if interaction.user.guild_permissions.manage_messages is False:
        await interaction.response.send_message("你沒有管理訊息的權限 | You don't have the permission to manage messages.", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)  # 延遲回應以保持 interaction 有效
    embed = Embed(
        title=f"正在刪除 {number} 則訊息 | Deleting {number} messages.",
        color=dc.Color.red()
    )
    embed.set_footer(text=f"Powered by HiHiYoyo606.")
    await interaction.followup.send(embed=embed, ephemeral=False)
    await asyncio.sleep(2)
    
    await interaction.channel.purge(limit=number+1)

    await send_new_info_logging(f"{interaction.user} has used /deletemessage with {number} messages deleted.")

@bot.tree.command(name="serverinfo", description="顯示伺服器資訊 | Show server information.")
async def slash_server_info(interaction: dc.Interaction):
    """顯示伺服器資訊"""
    """回傳: None"""
    if isinstance(interaction.channel, dc.DMChannel):
        await interaction.response.send_message("這個指令只能用在伺服器中 | This command can only be used in a server.", ephemeral=True)
        return
    
    server_name = interaction.guild.name
    member_count = interaction.guild.member_count
    owner = interaction.guild.owner
    create_at = interaction.guild.created_at.strftime("%Y-%m-%d")
    description = interaction.guild.description
    icon = interaction.guild.icon.url if interaction.guild.icon else None
    banner = interaction.guild.banner.url if interaction.guild.banner else None

    embed = dc.Embed(
        title="伺服器資訊 | Server Information",
        color=dc.Color.blue()
    )
    embed.set_thumbnail(url=icon)
    embed.add_field(name="伺服器名稱 | Server Name", value=server_name, inline=False)
    embed.add_field(name="成員數量 | Member Count", value=str(member_count), inline=False)
    embed.add_field(name="擁有者 | Owner", value=owner.mention, inline=False)
    embed.add_field(name="創建日期 | Created At", value=create_at, inline=False)
    embed.add_field(name="描述 | Description", value=description, inline=False)
    if banner:
        embed.set_image(url=banner)
    embed.set_footer(text=f"Powered by HiHiYoyo606.")

    await interaction.response.send_message(embed=embed, ephemeral=False)
    await send_new_info_logging(f"{interaction.user} has used /serverinfo to view server \"{server_name}\".")

@bot.tree.command(name="furinaphoto", description="顯示隨機一張芙寧娜的照片(每日搜尋額度有限請見諒) | Show a random photo of Furina.(Daily search limit exists)")
async def slash_furina_photo(interaction: dc.Interaction):
    """顯示隨機一張芙寧娜的照片"""
    """回傳: None"""
    # Defer the interaction publicly. We will edit this message later.
    await interaction.response.defer(thinking=True)
    await send_new_info_logging(f"{interaction.user} has used /furina_photo.")
    try:
        search_query = "芙寧娜" # Define the search term
        # Generate a random start index from the possible pages (1, 11, 21, ..., 91)
        possible_start_indices = [1 + i * 10 for i in range(10)] # Generates [1, 11, 21, ..., 91]
        random_start_index = random.choice(possible_start_indices)
        # Perform a single search with the random start index
        image_urls = await google_search(search_query, GOOGLE_SEARCH_API_KEY, GOOGLE_CSE_ID, num_results=10, start_index=random_start_index)

        if not image_urls:
            logging.warning(f"Google Image Search for '{search_query}' (start={random_start_index}) returned no results or failed.")
            # Edit the original deferred message to show the error
            await interaction.edit_original_response(content="抱歉，我找不到任何芙寧娜的照片！(網路搜尋失敗或沒有結果)")
            return
        # No need to shuffle if we only fetched one page's worth
        random_image_url = random.choice(image_urls)
        await send_new_info_logging("slash_furina_photo called, url returned: " + random_image_url)
        await interaction.edit_original_response(content=f"# 我可愛嗎:D | Am I cute?:D\n{random_image_url}")

    except Exception as e:
        # Log the error
        await send_new_error_logging(f"Error in slash_furina_photo: {e}")
        try:
            # Try to edit the original deferred message to show a generic error
            await interaction.edit_original_response(content="執行此指令時發生了內部錯誤，請稍後再試。")
        except dc.NotFound:
            # If editing fails, the interaction likely expired or was deleted
            await send_new_error_logging(f"Interaction expired or was deleted before sending error message for slash_furina_photo for {interaction.user}.")
        except dc.HTTPException as http_e:
             # Handle potential other HTTP errors during edit
             await send_new_error_logging(f"HTTP error editing interaction for slash_furina_photo error message: {http_e}")

# maybe music features

"""
@bot.tree.command(name="timeout", description="使一個用戶被停權(需擁有對成員停權權限) | Timeout a user in a text channel(Requires timeout members permission).")
@describe(user="要停權的用戶 | The user to be timed out.")
@describe(s="停權秒數 | The number of seconds to timeout.")
@describe(reason="停權原因 | The reason for timeout.")
async def text_mute(interaction: dc.Interaction, user: dc.Member, s: int, reason: str):
    if isinstance(interaction.channel, dc.DMChannel):
        await interaction.response.send_message("這個指令只能用在伺服器中 | This command can only be used in a server.", ephemeral=True)
        return
    if interaction.user.guild_permissions.moderate_members is False:
        await interaction.response.send_message("你沒有管理成員的權限 | You don't have the permission to manage members.", ephemeral=True)
        return
    
    await user.timeout(datetime.now() + timedelta(seconds=s), reason=reason)
    await interaction.response.send_message(f"# 水神的懲罰!! {user} 被停權 {s} 秒!! 原因: {reason}")
    send_new_info_logging(f"Someone is timed out.")
"""
    
@bot.event
async def on_ready():
    send_new_info_logging(f"Logged in as {bot.user}, system is ready.")

    try:
        synced = await bot.tree.sync()
        await send_new_info_logging(f"Synced {len(synced)} commands.")
    except Exception as e:
        await send_new_error_logging(f"Error syncing commands: {e}")

    await asyncio.sleep(3)  # 確保 WebSocket 初始化完成

@bot.event
async def on_message(message: dc.Message):
    await chat_process_message(message)  # 確保只執行一次

async def main():
    try:
        await bot.start(DISCORD_BOT_API_KEY)
        await send_new_info_logging(f"Bot successfully started at {get_hkt_time()}.", to_discord=False) 
    except dc.HTTPException as e:
        if e.status == 429:
            retry_after = e.response.headers.get("Retry-After")
            logging.warning(f"Rate limited! Retry after {retry_after} seconds.")
            # what? \
            await asyncio.sleep(delay=int(retry_after))
            return await main()

    except Exception as e:
        await send_new_error_logging(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
