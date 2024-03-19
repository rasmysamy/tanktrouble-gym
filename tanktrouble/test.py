
from pynput import keyboard
import matplotlib.pyplot as plt
import env.tanktrouble_env as tanktrouble

pressed = {'w': False, 's': False, 'a': False, 'd': False, 'up': False, 'down': False, 'left': False, 'right': False, 'fire': False}

plt.rcParams['keymap.save'].remove('s')

def on_press(key):
    if isinstance(key, keyboard.Key):
        return
    print(key.char, "pressed")
    pressed[key.char] = True


def on_release(key):
    if isinstance(key, keyboard.Key):
        if key is keyboard.Key.space:
            pressed['fire'] = True
        return
    print(key.char, "released")
    pressed[key.char] = False


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

env = tanktrouble.TankTrouble()
env.reset()

while True:
    plt.ion()
    plt.show()
    env.display_state(show=False)
    plt.draw()
    plt.pause(1.0 / 120.0)

    actions_1 = [pressed['w'], pressed['s'], pressed['a'], pressed['d'], pressed['fire']]
    pressed['fire'] = False

    # print(actions_1)

    observations, rewards, terminations, truncations, infos = env.step({"0": actions_1, "1": [False, False, False, False, False]})

    print(type(observations))
    print(observations)

    # print(terminations)

    if any(terminations.values()):
        env.reset()

    sleep(1)

    # print(env.p1_x)
    # print(env.p1_y)
    # print(env.p1_direction)

j = input()
