from models import *

rows = 4
cols = 4
sm = StateModel(rows, cols)
loc = Localizer(sm)

for move in range(5):
    print("move nbr:", move)
    loc.update()
