import env.tanktrouble_env as tanktrouble
import matplotlib.pyplot as plt

while True:
    env = tanktrouble.TankTrouble()
    env.reset()
    env.step(0)
    plt.ion()
    plt.show()
    env.display_state(show=False)
    plt.draw()
    plt.pause(1.0/120.0)

j = input()