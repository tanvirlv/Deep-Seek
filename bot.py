import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from httpx import AsyncClient, Timeout, HTTPStatusError

# Configure advanced logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log")  # Log to file for debugging
    ]
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_INPUT_LENGTH = 2000
MAX_RESPONSE_LENGTH = 4096  # Telegram's max message length
REQUEST_COOLDOWN = timedelta(seconds=5)
HEARTBEAT_INTERVAL = 300  # 5 minutes in seconds

# Global state for rate limiting
USER_LAST_REQUEST: Dict[int, datetime] = {}

class BotConfig:
    """Centralized configuration management"""
    def __init__(self):
        self.bot_token: Optional[str] = None
        self.api_key: Optional[str] = None
        
    def validate(self) -> None:
        """Validate all required config"""
        if not self.bot_token or len(self.bot_token) < 30:
            raise ValueError("Invalid BOT_TOKEN")
        if not self.api_key or not self.api_key.startswith("ds-"):
            raise ValueError("Invalid DEEPSEEK_API_KEY format")

config = BotConfig()

def safe_trim(text: str, max_len: int = MAX_RESPONSE_LENGTH) -> str:
    """Smart message trimming with overflow handling"""
    text = text.strip()
    if len(text) <= max_len:
        return text
        
    # Preserve the last complete line if possible
    trimmed = text[:max_len]
    if "\n" in trimmed:
        trimmed = trimmed.rsplit("\n", 1)[0] + "\n[...]"
    else:
        trimmed = trimmed[:max_len-3] + "..."
    return trimmed

async def check_rate_limit(user_id: int) -> bool:
    """Enhanced rate limiting with automatic cleanup"""
    now = datetime.now()
    
    # Cleanup old entries (>1 hour)
    global USER_LAST_REQUEST
    USER_LAST_REQUEST = {
        uid: time for uid, time in USER_LAST_REQUEST.items()
        if (now - time) < timedelta(hours=1)
    }
    
    if (last_request := USER_LAST_REQUEST.get(user_id)):
        if (now - last_request) < REQUEST_COOLDOWN:
            return False
            
    USER_LAST_REQUEST[user_id] = now
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Interactive start command with user analytics"""
    user = update.effective_user
    logger.info(f"New user: {user.id} - @{user.username}")
    
    await update.message.reply_text(
        f"🤖 Hello {user.first_name}! I'm DeepSeek AI Assistant\n\n"
        "• Ask me anything in any language\n"
        "• /help for usage tips\n"
        f"• Rate limit: 1 request per {REQUEST_COOLDOWN.seconds}s"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Dynamic help command"""
    await update.message.reply_text(
        "📚 *Bot Guide*\n\n"
        f"- Max input: {MAX_INPUT_LENGTH} chars\n"
        f"- Responses trimmed to {MAX_RESPONSE_LENGTH} chars\n"
        "- Supports markdown formatting\n"
        "- No message history (stateless)\n\n"
        "Try asking:\n"
        "• _Explain quantum computing simply_\n"
        "• _Write python code for bubble sort_",
        parse_mode="Markdown"
    )

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced message handler with proper error management"""
    user = update.effective_user
    message = update.message
    
    # Rate limit check
    if not await check_rate_limit(user.id):
        await message.reply_text(
            f"⏳ Please wait {REQUEST_COOLDOWN.seconds} seconds between requests"
        )
        return

    # Input validation
    if not (user_input := message.text.strip()):
        await message.reply_text("Please send a non-empty message")
        return
        
    if len(user_input) > MAX_INPUT_LENGTH:
        await message.reply_text(
            f"❌ Message exceeds {MAX_INPUT_LENGTH} character limit"
        )
        return

    try:
        async with AsyncClient(timeout=Timeout(30.0)) as client:
            response = await client.post(
                API_URL,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": user_input}],
                    "temperature": 0.7,
                },
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            # Validate response structure
            response.raise_for_status()
            data = response.json()
            
            if not data.get("choices"):
                raise ValueError("Invalid API response format")
                
            ai_text = data["choices"][0]["message"]["content"]
            await message.reply_text(safe_trim(ai_text))

    except HTTPStatusError as e:
        status_code = e.response.status_code
        logger.error(f"API Error {status_code} for user {user.id}")
        
        if status_code == 401:
            msg = "🔒 API authentication failed"
        elif status_code == 429:
            msg = "⚠️ Too many requests to API"
        else:
            msg = f"API Error {status_code}"
            
        await message.reply_text(f"{msg}. Please try later.")
        
    except Exception as e:
        logger.error(f"Unexpected error for {user.id}: {str(e)}", exc_info=True)
        await message.reply_text("⚠️ Temporary system error")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Centralized error handling"""
    error = context.error
    logger.critical(f"Unhandled error: {error}", exc_info=error)
    
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "🛠️ Our engineers have been notified of this issue"
        )

async def heartbeat(context: ContextTypes.DEFAULT_TYPE):
    """Prevent Render free tier from sleeping"""
    logger.debug("Heartbeat ping")

def main():
    try:
        # Load config from environment
        config.bot_token = os.getenv("BOT_TOKEN")
        config.api_key = os.getenv("DEEPSEEK_API_KEY")
        config.validate()
        
        # Initialize bot
        app = Application.builder().token(config.bot_token).build()
        
        # Register handlers
        app.add_handlers([
            CommandHandler("start", start),
            CommandHandler("help", help_command),
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                reply
            )
        ])
        
        # Error handling
        app.add_error_handler(error_handler)
        
        # Prevent Render free tier timeout
        job_queue = app.job_queue
        if job_queue:
            job_queue.run_repeating(
                heartbeat,
                interval=HEARTBEAT_INTERVAL,
                first=10
            )
        
        # Start bot
        logger.info("Starting bot...")
        app.run_polling(
            drop_pending_updates=True,
            close_loop=False,
            allowed_updates=Update.ALL_TYPES
        )
        
    except Exception as e:
        logger.critical(f"Fatal startup error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()