import logging
import multiprocessing
from AI.vei import Vei
from models.card_registry import CardRegistry
from scripts_of_tribute.game import Game

if __name__ == "__main__":
    bot = Vei(bot_name="Vei")
    game = Game()
    game.register_bot(bot)
    game.run(
        "Vei", "RandomBot",
        start_game_runner=True
    )