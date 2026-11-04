"""Entry point: wires bot.py's handlers into channel/discord and starts it.

Run from the project root with: uv run python main.py
"""

import dotenv

dotenv.load_dotenv()

import bot
from channel import discord as discord_channel

discord_channel.on_prompt = bot.handle_prompt
discord_channel.on_startup = bot.handle_startup


def main():
    discord_channel.run()


if __name__ == "__main__":
    main()
