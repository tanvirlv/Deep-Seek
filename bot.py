import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from httpx import AsyncClient, Timeout, HTTPStatusError, Limits

# Configure advanced logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding='utf-8', delay=True)  # Lazy file opening
    ]
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_INPUT_LENGTH = 2000
MAX_RESPONSE_LENGTH = 4096
REQUEST_COOLDOWN = timedelta(seconds=5)
HEARTBEAT_INTERVAL = 300  # 5 minutes
MAX_API_RETRIES = 3
API_RETRY_DELAY = 1.5

class BotConfig:
    """Secure configuration management with strict validation"""
    def __init__(self):
        self.bot_token: Optional[str] = None
        self.api_key: Optional[str] = None
        
    def validate(self) -> Tuple[bool, str]:
        """Returns (is_valid, error_message)"""
        if not self.bot_token or not (
            len(self.bot_token) > 30 
            and ':' in self.bot_token
            and self.bot_token.split(':')[0].isdigit()
        ):
            return False, "Invalid Telegram bot token format"
            
        if not self.api_key or not (
            self.api_key.startswith(("sk-", "sk-or-")) 
            and len(self.api_key) > 50
        ):
            return False, "API key must start with 'sk-' or 'sk-or-' and be >50 chars"
            
        return True, ""

config = BotConfig()

def safe_trim(text: str, max_len: int = MAX_RESPONSE_LENGTH) -> str:
    """Smarter message truncation preserving markdown"""
    text = text.strip()
    if len(text) <= max_len:
        return text
        
    # Priority preservation
    if "```" in text[:max_len]:  # Code blocks
        return text[:max_len] + "\n[...]"
    if "\n\n" in text[:max_len-10]:  # Paragraphs
        return text[:max_len].rsplit("\n\n", 1)[0] + "\n[...]"
    return text[:max_len-3] + "..."

class RateLimiter:
    """Thread-safe rate limiting with automatic cleanup"""
    _instance = None
    user_requests: Dict[int, datetime] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    async def check_rate_limit(self, user_id: int) -> bool:
        async with self._lock:
            now = datetime.now()
            # Clean old entries (>1 hour)
            self.user_requests = {
                uid: t for uid, t in self.user_requests.items() 
                if (now - t) < timedelta(hours=1)
            }
            
            if (last := self.user_requests.get(user_id)):
                if (now - last) < REQUEST_COOLDOWN:
                    return False
                    
            self.user_requests[user_id] = now
            return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced start command with user analytics"""
    user = update.effective_user
    logger.info(f"New session: {user.id} | @{user.username or 'no-username'}")
    
    await update.message.reply_text(
        f"👋 Welcome *{user.first_name or 'there'}*!\n\n"
        "🚀 *DeepSeek AI Assistant*\n"
        "• Ask complex questions\n"
        "• Get code examples\n"
        "• Explain concepts\n\n"
        "Try:\n"
        "• `/help` for guidance\n"
        "• `Explain like I'm 5`\n"
        "• `Python fibonacci code`",
        parse_mode="Markdown",
        disable_web_page_preview=True
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Interactive help with formatted examples"""
    await update.message.reply_text(
        "📚 *Bot Guide*\n\n"
        f"🔹 *Message Limit:* {MAX_INPUT_LENGTH} chars\n"
        f"🔹 *Cooldown:* {REQUEST_COOLDOWN.seconds}s\n\n"
        "💡 *Examples:*\n"
        "• `Summarize quantum physics`\n"
        "• `Write a Python REST API`\n"
        "• `Debug this error: ...`\n\n"
        "⚙️ *Commands:* `/start` `/help`",
        parse_mode="Markdown"
    )

async def call_deepseek_api(prompt: str) -> str:
    """Robust API caller with retry logic"""
    last_error = ""
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            async with AsyncClient(
                timeout=Timeout(30.0),
                limits=Limits(max_connections=1)
            ) as client:
                response = await client.post(
                    API_URL,
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": min(MAX_RESPONSE_LENGTH, 2000)
                    },
                    headers={
                        "Authorization": f"Bearer {config.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("choices"):
                    raise ValueError("Empty choices in response")
                    
                return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_API_RETRIES:
                await asyncio.sleep(API_RETRY_DELAY * attempt)
            else:
                raise Exception(f"API failed after {MAX_API_RETRIES} attempts: {last_error}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main message handler with full error protection"""
    user = update.effective_user
    message = update.message
    
    # Rate limiting
    limiter = RateLimiter()
    if not await limiter.check_rate_limit(user.id):
        await message.reply_text(
            f"⏳ Please wait {REQUEST_COOLDOWN.seconds}s between requests",
            reply_to_message_id=message.message_id
        )
        return

    # Input validation
    user_input = message.text.strip()
    if not user_input:
        await message.reply_text("Please send a non-empty message")
        return
        
    if len(user_input) > MAX_INPUT_LENGTH:
        await message.reply_text(
            f"❌ Max {MAX_INPUT_LENGTH} chars allowed",
            reply_to_message_id=message.message_id
        )
        return

    try:
        reply_text = await call_deepseek_api(user_input)
        await message.reply_text(
            safe_trim(reply_text),
            parse_mode="Markdown",
            reply_to_message_id=message.message_id
        )

    except HTTPStatusError as e:
        error_map = {
            401: "🔑 Invalid API key - contact admin",
            429: "🚦 Too many requests - try later",
            500: "🌐 API server error"
        }
        error_msg = error_map.get(e.response.status_code, 
                               f"API Error {e.response.status_code}")
        logger.error(f"API fail {user.id}: {e.response.status_code}")
        await message.reply_text(error_msg)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        await message.reply_text(
            "⚠️ Temporary error - developers notified",
            reply_to_message_id=message.message_id
        )

async def heartbeat(context: ContextTypes.DEFAULT_TYPE):
    """Keep-alive for Render free tier"""
    try:
        await context.bot.get_me()  # Active check
        logger.debug("♥ Heartbeat OK")
    except Exception as e:
        logger.error(f"Heartbeat failed: {str(e)}")

async def post_init(app: Application) -> None:
    """Validate setup before starting"""
    try:
        await app.bot.get_me()
        logger.info("✅ Bot validated")
    except Exception as e:
        logger.critical(f"Startup validation failed: {str(e)}")
        raise

def setup_application() -> Application:
    """Configure the bot application"""
    # Load config
    config.bot_token = os.getenv("BOT_TOKEN")
    config.api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # Validate
    is_valid, error_msg = config.validate()
    if not is_valid:
        logger.critical(f"Config error: {error_msg}")
        raise ValueError(error_msg)

    # Initialize
    app = Application.builder() \
        .token(config.bot_token) \
        .post_init(post_init) \
        .build()
    
    # Handlers
    app.add_handlers([
        CommandHandler("start", start),
        CommandHandler("help", help_command),
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    ])
    
    app.add_error_handler(error_handler)
    
    # Heartbeat
    if app.job_queue:
        app.job_queue.run_repeating(
            heartbeat,
            interval=HEARTBEAT_INTERVAL,
            first=10
        )
    
    return app

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler"""
    error = context.error
    logger.critical(f"UNHANDLED ERROR: {error}", exc_info=error)
    
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "⚡ System error - please try again later",
            reply_to_message_id=update.message.message_id
        )

def main():
    try:
        logger.info("🚀 Starting bot...")
        app = setup_application()
        
        # Mask sensitive info in logs
        logger.info(
            f"Config loaded - "
            f"Bot: {config.bot_token[:5]}...{config.bot_token[-2:]}, "
            f"API: {config.api_key[:5]}..."
        )
        
        app.run_polling(
            drop_pending_updates=True,
            close_loop=True,
            allowed_updates=Update.ALL_TYPES
        )
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped manually")
    except Exception as e:
        logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("🧹 Cleaning up resources...")

if __name__ == "__main__":
    main()