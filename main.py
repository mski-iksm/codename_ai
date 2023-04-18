from codename_ai.model.boss import FastTextBossModel
from codename_ai.model.game import Game

game = Game.setup_game(random_seed=111)
model = FastTextBossModel.setup_model('blue')

model.next_hint(game=game)