import time
import timeit
from time import sleep

from pynput import keyboard
import matplotlib.pyplot as plt
import env.tanktrouble_env as tanktrouble

pressed = {'w': False, 's': False, 'a': False, 'd': False, 'up': False, 'down': False, 'left': False, 'right': False,
           'fire': False, 'enter': False}

plt.rcParams['keymap.save'].remove('s')


def on_press(key):
    if isinstance(key, keyboard.Key):
        if key is keyboard.Key.up:
            pressed['up'] = True
        elif key is keyboard.Key.down:
            pressed['down'] = True
        elif key is keyboard.Key.left:
            pressed['left'] = True
        elif key is keyboard.Key.right:
            pressed['right'] = True
        return
    print(key.char, "pressed")
    pressed[key.char] = True


def on_release(key):
    if isinstance(key, keyboard.Key):
        if key is keyboard.Key.space:
            pressed['fire'] = True
        elif key is keyboard.Key.enter:
            pressed['enter'] = True
        elif key is keyboard.Key.up:
            pressed['up'] = False
        elif key is keyboard.Key.down:
            pressed['down'] = False
        elif key is keyboard.Key.left:
            pressed['left'] = False
        elif key is keyboard.Key.right:
            pressed['right'] = False
        return
    print(key.char, "released")
    pressed[key.char] = False


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

env = tanktrouble.TankTrouble()
env.reset()

frame_rate = 60

last_time = -1


def loop():
    global last_time
    # plt.ion()
    # plt.show()
    # env.display_state(show=False)
    # plt.draw()
    # plt.pause(1.0 / 120.0)]
    # if time.time() - last_time >= 1.0 / frame_rate:
    #     last_time = time.time()
    #     env.pygame_render()
    env.pygame_render()

    actions_1 = [pressed['w'], pressed['s'], pressed['a'], pressed['d'], pressed['fire']]
    actions_2 = [pressed['up'], pressed['down'], pressed['left'], pressed['right'], pressed['enter']]
    pressed['fire'] = False
    pressed['enter'] = False

    # print(actions_1)

    observations, rewards, terminations, truncations, infos = env.step({"0": actions_1, "1": actions_2})

    # print(type(observations))
    # print(observations)

    # print(terminations)

    if any(terminations.values()):
        env.reset()

    sleep(1.0 / frame_rate)

    # print(env.p1_x)
    # print(env.p1_y)
    # print(env.p1_direction)


print(timeit.timeit(loop, number=100_00), "seconds")
exit(0)

while True:
    # use timeit to benchmark loop
    loop()

# j = input()
