#!/usr/bin/env python3
"""Main entry point for MCPulse application."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui import create_app
from src.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcpulse.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    logger.info("Starting MCPulse application")
    logger.info(f"Gradio server will run on {settings.gradio_server_name}:{settings.gradio_server_port}")
    
    # Check for API keys
    if not settings.openai_api_key and not settings.anthropic_api_key:
        logger.warning("⚠️  No LLM API key configured!")
        logger.warning("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        logger.warning("The application will start, but AI features will be limited.")
    
    try:
        # Create and launch Gradio app
        app = create_app()
        
        app.launch(
            server_name=settings.gradio_server_name,
            server_port=settings.gradio_server_port,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("MCPulse application stopped")


if __name__ == "__main__":
    main()
