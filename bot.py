import os
import logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from httpx import AsyncClient, Timeout, HTTPStatusError
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_INPUT_LENGTH = 2000
MAX_RESPONSE_LENGTH = 4096
REQUEST_COOLDOWN = timedelta(seconds=5)

# Global state for rate limiting
USER_LAST_REQUEST: Dict[int, datetime] = {}

class ConfigError(Exception):
    """Custom exception for configuration errors"""

def validate_config() -> Tuple[str, str]:
    """Validate and return environment variables"""
    config = {
        "BOT_TOKEN": (str, 30, 100),  # (type, min_len, max_len)
        "DEEPSEEK_API_KEY": (str, 20, 200),
    }

    validated = {}
    for var, (var_type, min_l, max_l) in config.items():
        value = os.getenv(var)
        if not value:
            raise ConfigError(f"{var} environment variable is not set")
        if not isinstance(value, var_type):
            raise ConfigError(f"{var} must be {var_type.__name__}")
        if not (min_l <= len(value) <= max_l):
            raise ConfigError(f"{var} length must be between {min_l}-{max_l} chars")
        validated[var] = value

    return validated["BOT_TOKEN"], validated["DEEPSEEK_API_KEY"]

def safe_trim(text: str, max_len: int = MAX_RESPONSE_LENGTH) -> str:
    """Safely trim long messages at the nearest line break"""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit("\n", 1)[0] + "\n[...truncated]"

async def check_rate_limit(user_id: int) -> bool:
    """Return True if request should be allowed"""
    now = datetime.now()
    last_request = USER_LAST_REQUEST.get(user_id)
    
    if last_request and (now - last_request) < REQUEST_COOLDOWN:
        return False
        
    USER_LAST_REQUEST[user_id] = now
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🤖 I am DeepSeek V3 0348 chat bot created by tanvir_lv!\n\n"
        "• Just send me a message\n"
        "• Use /help for more info\n"
        "• Rate limit: 1 request per 5 seconds"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    await update.message.reply_text(
        "ℹ️ Bot Usage Guide:\n\n"
        f"- Keep messages under {MAX_INPUT_LENGTH} chars\n"
        "- I process text only (no files/images)\n"
        "- Responses limited to {MAX_RESPONSE_LENGTH} chars\n"
        "- Cooldown: {REQUEST_COOLDOWN.seconds} sec between requests"
    )

def validate_api_response(response_data: dict) -> str:
    """Validate and extract response text from API"""
    if not isinstance(response_data, dict):
        raise ValueError("API response is not a dictionary")
    
    if "choices" not in response_data or not isinstance(response_data["choices"], list):
        raise ValueError("Invalid choices format")
        
    first_choice = response_data["choices"][0]
    if "message" not in first_choice or "content" not in first_choice["message"]:
        raise ValueError("Missing message content")
        
    return str(first_choice["message"]["content"])

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process user messages"""
    user = update.effective_user
    message = update.message
    
    # Rate limiting check
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
            f"Message too long. Please keep under {MAX_INPUT_LENGTH} characters"
        )
        return

    try:
        async with AsyncClient(timeout=Timeout(30.0)) as client:
            response = await client.post(
                API_URL,
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": user_input}],
                },
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            )
            response.raise_for_status()
            
            ai_text = validate_api_response(response.json())
            await message.reply_text(safe_trim(ai_text))

    except Timeout:
        logger.warning(f"Timeout for user {user.id}")
        await message.reply_text("⌛ Server is busy. Please try again later")
    except HTTPStatusError as e:
        logger.error(f"API error for user {user.id}: {e.response.status_code}")
        await message.reply_text("⚠️ Service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error for user {user.id}: {str(e)}", exc_info=True)
        await message.reply_text("🔧 An error occurred. Developers have been notified")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle all uncaught exceptions"""
    logger.error(
        f"Error during processing update {update}: {context.error}",
        exc_info=context.error,
    )
    
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "⚠️ A system error occurred. Please try again later"
        )

def main():
    """Start the bot"""
    try:
        global BOT_TOKEN, DEEPSEEK_API_KEY
        BOT_TOKEN, DEEPSEEK_API_KEY = validate_config()
        
        app = Application.builder().token(BOT_TOKEN).build()
        
        # Register handlers
        app.add_handlers([
            CommandHandler("start", start),
            CommandHandler("help", help_command),
            MessageHandler(filters.TEXT & ~filters.COMMAND, reply),
        ])
        
        app.add_error_handler(error_handler)
        
        logger.info("Bot is starting...")
        app.run_polling(drop_pending_updates=True)
        
    except Exception as e:
        logger.critical(f"Fatal startup error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()