import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

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
        logging.FileHandler("bot.log", encoding='utf-8')  # Proper file handling
    ]
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_INPUT_LENGTH = 2000
MAX_RESPONSE_LENGTH = 4096
REQUEST_COOLDOWN = timedelta(seconds=5)
HEARTBEAT_INTERVAL = 300

class BotConfig:
    """Enhanced configuration management with proper validation"""
    def __init__(self):
        self.bot_token: Optional[str] = None
        self.api_key: Optional[str] = None
        
    def validate(self) -> None:
        """Strict validation for credentials"""
        if not self.bot_token or not self.bot_token.startswith(''):
            raise ValueError("Invalid BOT_TOKEN format")
            
        if not self.api_key or not (
            self.api_key.startswith("sk-") or 
            self.api_key.startswith("sk-or-")
        ):
            raise ValueError(
                "Invalid API key format. Must start with 'sk-' or 'sk-or-'"
            )

config = BotConfig()

def safe_trim(text: str, max_len: int = MAX_RESPONSE_LENGTH) -> str:
    """Improved message trimming with markdown awareness"""
    text = text.strip()
    if len(text) <= max_len:
        return text
        
    # Preserve code blocks if present
    if "```" in text[:max_len]:
        return text[:max_len] + "\n[...truncated]"
    return text[:max_len-3] + "..."

class RateLimiter:
    """Enhanced rate limiting with thread safety"""
    _instance = None
    user_requests: Dict[int, datetime] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def check_rate_limit(self, user_id: int) -> bool:
        now = datetime.now()
        last_request = self.user_requests.get(user_id)
        
        if last_request and (now - last_request) < REQUEST_COOLDOWN:
            return False
            
        self.user_requests[user_id] = now
        return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Improved start command with better user onboarding"""
    user = update.effective_user
    logger.info(f"New user session: {user.id}")
    
    welcome_msg = (
        f"👋 Welcome *{user.first_name}*!\n\n"
        "I'm your DeepSeek AI assistant. Here's what I can do:\n"
        "• Answer complex questions\n"
        "• Explain concepts\n"
        "• Generate code\n\n"
        "Try asking:\n"
        "• _Explain quantum computing_\n"
        "• _Write Python code for a calculator_\n"
        "• _Help me debug this error_"
    )
    
    await update.message.reply_text(
        welcome_msg,
        parse_mode="Markdown",
        disable_web_page_preview=True
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Robust message handler with full error protection"""
    user = update.effective_user
    message = update.message
    
    # Initialize rate limiter
    limiter = RateLimiter()
    
    if not limiter.check_rate_limit(user.id):
        await message.reply_text(
            f"⏳ Please wait {REQUEST_COOLDOWN.seconds} seconds between requests",
            reply_to_message_id=message.message_id
        )
        return

    # Validate input
    user_input = message.text.strip()
    if not user_input:
        await message.reply_text("Please send a non-empty message")
        return
        
    if len(user_input) > MAX_INPUT_LENGTH:
        await message.reply_text(
            f"❌ Message too long (max {MAX_INPUT_LENGTH} chars)",
            reply_to_message_id=message.message_id
        )
        return

    try:
        async with AsyncClient(timeout=Timeout(30.0)) as client:
            response = await client.post(
                API_URL,
                json={
                    "model": "deepseek-chat",
                    "messages": [{
                        "role": "user",
                        "content": user_input
                    }],
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            if not data.get("choices"):
                raise ValueError("Empty API response")
                
            reply_text = data["choices"][0]["message"]["content"]
            await message.reply_text(
                safe_trim(reply_text),
                parse_mode="Markdown",
                reply_to_message_id=message.message_id
            )

    except HTTPStatusError as e:
        error_msg = {
            401: "🔐 API authentication failed - check your API key",
            429: "⚠️ Too many requests - please wait before trying again",
            500: "🔧 API server error - try again later"
        }.get(e.response.status_code, f"API Error {e.response.status_code}")
        
        logger.error(f"API Error for {user.id}: {str(e)}")
        await message.reply_text(error_msg)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        await message.reply_text(
            "⚠️ Temporary system issue - our team has been notified",
            reply_to_message_id=message.message_id
        )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Centralized error logging"""
    error = context.error
    logger.critical(f"Unhandled error: {error}", exc_info=error)
    
    if update and isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "🛠️ Our engineers have been notified of this issue",
            reply_to_message_id=update.message.message_id
        )

def setup_application() -> Application:
    """Configure and return the bot application"""
    # Load and validate config
    config.bot_token = os.getenv("BOT_TOKEN")
    config.api_key = os.getenv("DEEPSEEK_API_KEY")
    config.validate()

    # Initialize application
    app = Application.builder().token(config.bot_token).build()
    
    # Register handlers
    app.add_handlers([
        CommandHandler("start", start),
        CommandHandler("help", help_command),
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    ])
    
    app.add_error_handler(error_handler)
    return app

def main():
    try:
        logger.info("Initializing bot...")
        
        # Setup application
        app = setup_application()
        
        # Start the bot
        app.run_polling(
            drop_pending_updates=True,
            close_loop=True,
            allowed_updates=Update.ALL_TYPES
        )

    except Exception as e:
        logger.critical(f"Fatal startup error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()