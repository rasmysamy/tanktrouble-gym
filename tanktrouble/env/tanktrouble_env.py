import math

import gymnasium.spaces.utils
import pygame
from pettingzoo import ParallelEnv

import functools
import random
# import jax
# import chex

from copy import copy, deepcopy
from math import sqrt

import matplotlib.pyplot as plt

from gymnasium.spaces import Discrete, MultiDiscrete, Box, Tuple, MultiBinary
import numpy as np

env_id = 0

fixed_walls = True

image_in_obs = True

class Ball:
    def __init__(self, x=-100.0, y=-100.0, vx=-100.0, vy=-100.0, life=-1):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life

    def return_state(self):
        return self.x, self.y, self.vx, self.vy


# @chex.dataclass
# class ball:
#     x: float
#     y: float
#     vx: float
#     vy: float
#     life: int
#
#     def return_state(self):
#         return self.x, self.y, self.vx, self.vy
#
# class player:


class TankTrouble(ParallelEnv):
    metadata = {
        "name": "tanktrouble_v0",
    }

    def draw_line(self, p1, p2, color):
        pygame.draw.line(self.pygame_scene, color, self.scale((p1[0], p1[1])), self.scale((p2[0], p2[1])), 3)

    def draw_circle(self, p, color):
        pygame.draw.circle(self.pygame_scene, color, self.scale((p[0], p[1])), 5)

    def __init__(self, s_x=8, s_y=5):
        self.seen_cells = {"0": set(), "1": set()}
        self.always_render = False
        self.pygame_scene = None
        self.scale = lambda x: [x[0] / self.size_x * 640, (1 - x[1] / self.size_y) * 480]
        self.mask = np.ones(5, dtype=bool)

        global env_id
        self.env_id = env_id
        env_id += 1
        self.agents = ["0", "1"]
        self.tank_length = 0.5
        self.tank_width = 0.3
        self.tank_speed = 0.05
        self.size_x = s_x
        self.size_y = s_y
        self.p1_x = None
        self.p1_y = None
        self.p2_x = None
        self.p2_y = None
        self.p1_direction = None
        self.p2_direction = None
        self.remaining_time = 10000
        self.possible_agents = ["0", "1"]
        self.ball_speed = self.tank_speed * 2
        self.max_speed = 5.0
        self.p_wall = 0.4
        self.ball_lifetime = 200
        self.action_transform = lambda x: x
        self.action_untransform = lambda x: x

        self.min_start_dist = max(self.size_x, self.size_y) / 1.33

        self.p1_v = 0.0
        self.p2_v = 0.0

        self.agent_name_mapping = {"0": "p1", "1": "p2"}
        self.max_balls = 5
        self.p1_balls = [Ball() for _ in range(self.max_balls)]
        self.p2_balls = [Ball() for _ in range(self.max_balls)]
        self.horizontal_walls = [[_ for _ in range(self.size_y + 1)] for _ in range(self.size_x)]
        self.vertical_walls = [[_ for _ in range(self.size_y)] for _ in range(self.size_x + 1)]
        self.action_spaces = dict()
        self.observation_spaces = dict()
        self.action_spaces["0"] = MultiBinary(5)
        self.action_spaces["1"] = MultiBinary(5)

        obs_space_noimg = [MultiBinary([self.size_x + 1, self.size_y + 1]), MultiBinary([self.size_x + 1, self.size_y + 1]),
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # self state
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # other state
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(4, self.max_balls)),  # own balls
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(4, self.max_balls)),  # other balls
             Box(low=0, high=self.remaining_time, shape=(1, self.max_balls)),  # own balls remaining life
             Box(low=0, high=self.remaining_time, shape=(1, self.max_balls)),  # other balls remaining life
             MultiBinary([self.max_balls]),  # own balls valid
             MultiBinary([self.max_balls]),  # other balls valid
             Box(low=0, high=self.remaining_time, shape=(1,))]

        self.img_size = (self.size_x * 3, self.size_y * 3, 5)

        obs_space_img = Tuple([Box(low=0, high=255, shape=self.img_size)] + obs_space_noimg)
        obs_space = obs_space_img if image_in_obs else Tuple(obs_space_noimg)

        self.observation_spaces["0"] = obs_space
        self.observation_spaces["1"] = obs_space
        self.observation_shape = gymnasium.spaces.flatdim(self.observation_spaces["0"])
        self.action_shape = gymnasium.spaces.flatdim(self.action_spaces["0"])

    def num_agents(self) -> int:
        return 2

    def max_num_agents(self) -> int:
        return 2

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def is_path(self, x1, y1, x2, y2):
        x1 = math.floor(x1)
        y1 = math.floor(y1)
        x2 = math.floor(x2)
        y2 = math.floor(y2)
        # we do a bfs starting from x1, y1 and check if we can reach x2, y2
        visited = [[False for _ in range(self.size_y)] for _ in range(self.size_x)]
        queue = [(x1, y1)]
        visited[x1][y1] = True
        while len(queue) > 0:
            x, y = queue.pop(0)
            if x == x2 and y == y2:
                return True
            if x > 0 and not self.vertical_walls[x][y] and not visited[x - 1][y]:
                visited[x - 1][y] = True
                queue.append((x - 1, y))
            if x < self.size_x and not self.vertical_walls[x + 1][y] and not visited[x + 1][y]:
                visited[x + 1][y] = True
                queue.append((x + 1, y))
            if y > 0 and not self.horizontal_walls[x][y] and not visited[x][y - 1]:
                visited[x][y - 1] = True
                queue.append((x, y - 1))
            if y < self.size_y and not self.horizontal_walls[x][y + 1] and not visited[x][y + 1]:
                visited[x][y + 1] = True
                queue.append((x, y + 1))
        return False

    def reset(self, seed=None, options=None):
        global fixed_walls
        self.seen_cells = {"0": set(), "1": set()}
        self.agents = copy(self.possible_agents)

        while True:
            self.p1_x = random.randint(0, self.size_x - 1) + 0.5
            self.p1_y = random.randint(0, self.size_y - 1) + 0.5
            self.p2_x = random.randint(0, self.size_x - 1) + 0.5
            self.p2_y = random.randint(0, self.size_y - 1) + 0.5
            if sqrt((self.p1_x - self.p2_x) ** 2 + (self.p1_y - self.p2_y) ** 2) > self.min_start_dist:
                break

        self.p1_direction = random.random() * 2 * math.pi
        self.p2_direction = random.random() * 2 * math.pi
        self.p1_v = 0.0
        self.p2_v = 0.0
        self.remaining_time = 1000
        self.p1_balls = [Ball() for _ in range(self.max_balls)]
        self.p2_balls = [Ball() for _ in range(self.max_balls)]
        self.agents = ["0", "1"]

        while True:

            # self.horizontal_walls = [[random.random() - (1 if idx in [0, self.size_y] else 0) < self.p_wall for idx in
            #                           range(self.size_y + 1)] for _ in
            #                          range(self.size_x + 1)]  # outside walls are always there
            # self.vertical_walls = [
            #     [random.random() - (1 if idx in [0, self.size_x] else 0) < self.p_wall for _ in range(self.size_y + 1)]
            #     for
            #     idx in range(self.size_x + 1)]  # outside walls are always there
            # # check if there is a path from p1 to p2

            if fixed_walls:
                self.vertical_walls = [[True, True, True, True, True, True], [True, True, False, False, False, False],
                                      [True, False, True, True, True, True], [True, True, False, False, True, True],
                                      [True, False, False, False, False, False], [False, False, True, False, False, True],
                                      [False, True, True, False, False, False], [False, False, True, False, False, False],
                                      [True, True, True, True, True, True]]
                self.horizontal_walls = [[True, True, True, False, False, True], [True, False, False, True, True, True],
                                        [True, False, False, True, True, True], [True, False, True, False, False, True],
                                        [True, False, False, False, False, True], [True, False, False, True, True, True],
                                        [True, False, True, False, True, True], [True, True, False, False, False, True],
                                        [True, False, True, False, False, True]]
            else:
                self.horizontal_walls = [[random.random() - (1 if idx in [0, self.size_y] else 0) < self.p_wall for idx in
                                          range(self.size_y + 1)] for _ in
                                         range(self.size_x + 1)]  # outside walls are always there
                self.vertical_walls = [
                    [random.random() - (1 if idx in [0, self.size_x] else 0) < self.p_wall for _ in range(self.size_y + 1)]
                    for
                    idx in range(self.size_x + 1)]  # outside walls are always there
                # check if there is a path from p1 to p2




            while True:
                self.p1_x = random.randint(0, self.size_x - 1) + 0.5
                self.p1_y = random.randint(0, self.size_y - 1) + 0.5
                self.p2_x = random.randint(0, self.size_x - 1) + 0.5
                self.p2_y = random.randint(0, self.size_y - 1) + 0.5
                if sqrt((self.p1_x - self.p2_x) ** 2 + (self.p1_y - self.p2_y) ** 2) > self.min_start_dist:
                    break

            if self.is_path(self.p1_x, self.p1_y, self.p2_x, self.p2_y):
                break

        return self.get_obs(), {"0": {}, "1": {}}

    def display_state(self, show=True):
        # we draw the state using matplotlib
        plt.clf()
        plt.xlim(0, self.size_x)
        plt.ylim(0, self.size_y)
        self.draw_tank(self.p1_x, self.p1_y, self.p1_direction, 'r')
        self.draw_tank(self.p2_x, self.p2_y, self.p2_direction, 'b')
        plt.plot(self.p1_x, self.p1_y, 'ro')
        plt.plot(self.p2_x, self.p2_y, 'bo')
        plt.plot([self.p1_x, self.p1_x + 0.5 * math.cos(self.p1_direction)],
                 [self.p1_y, self.p1_y + 0.5 * math.sin(self.p1_direction)], 'r')
        plt.plot([self.p2_x, self.p2_x + 0.5 * math.cos(self.p2_direction)],
                 [self.p2_y, self.p2_y + 0.5 * math.sin(self.p2_direction)], 'b')
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.horizontal_walls[i][j]:
                    plt.plot([i, i + 1], [j, j], 'k')
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.vertical_walls[i][j]:
                    plt.plot([i, i], [j, j + 1], 'k')
        for ball in self.p1_balls:
            if ball.life > 0:
                plt.plot(ball.x, ball.y, 'r*')
        for ball in self.p2_balls:
            if ball.life > 0:
                plt.plot(ball.x, ball.y, 'b*')
        if show:
            plt.draw()
            plt.pause(1.0 / 30.0)
            plt.show()

    def copy(self):
        surf = self.pygame_scene
        self.pygame_scene = None
        self.__deepcopy___ = self.__deepcopy__
        self.__deepcopy__ = None
        cp = deepcopy(self)
        self.__deepcopy__ = self.__deepcopy___
        cp.__deepcopy__ = cp.__deepcopy___
        self.pygame_scene = surf
        cp.pygame_scene = pygame.Surface((640, 480))
        return cp

    def __deepcopy__(self, memodict={}):
        cp = self.copy()
        memodict[id(self)] = cp
        return cp

    def pygame_render(self):
        if self.pygame_scene is None:
            self.pygame_scene = pygame.Surface((640, 480))
        # we do it in pygame instead of matplotlib
        self.pygame_scene.fill((255, 255, 255))
        self.draw_tank_pygame(self.p1_x, self.p1_y, self.p1_direction, (255, 0, 0))
        self.draw_tank_pygame(self.p2_x, self.p2_y, self.p2_direction, (0, 0, 255))
        pygame.draw.circle(self.pygame_scene, (255, 0, 0), (int(self.p1_x), int(self.p1_y)), 5)
        pygame.draw.circle(self.pygame_scene, (0, 0, 255), (int(self.p2_x), int(self.p2_y)), 5)
        self.draw_line([self.p1_x, self.p1_y],
                       [self.p1_x + 0.5 * math.cos(self.p1_direction), self.p1_y + 0.5 * math.sin(self.p1_direction), ],
                       (255, 0, 0))
        self.draw_line([self.p2_x, self.p2_y],
                       [self.p2_x + 0.5 * math.cos(self.p2_direction), self.p2_y + 0.5 * math.sin(self.p2_direction)],
                       (0, 0, 255))
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.horizontal_walls[i][j]:
                    self.draw_line([i, j], [i + 1, j], (0, 0, 0))
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.vertical_walls[i][j]:
                    self.draw_line([i, j], [i, j + 1], (0, 0, 0))
        for ball in self.p1_balls:
            if ball.life > 0:
                self.draw_circle([ball.x, ball.y], (255, 0, 0))
        for ball in self.p2_balls:
            if ball.life > 0:
                self.draw_circle([ball.x, ball.y], (0, 0, 255))
        if not pygame.display.get_init():
            pygame.display.init()
            pygame.display.set_caption("TankTrouble")
            pygame.display.set_mode((640, 480))
        pygame.display.get_surface().blit(self.pygame_scene, (0, 0))
        pygame.display.flip()

    def draw_tank(self, x, y, direction, color):

        # draw a rectangle at x, y with the direction
        rect = plt.Rectangle((x - self.tank_length / 2, y - self.tank_width / 2), self.tank_length, self.tank_width,
                             angle=direction * 180 / math.pi, color=color, rotation_point=(x, y))

        px = x
        py = y
        rot = direction

        c1x = px + (self.tank_length / 2) * math.cos(rot) + (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c1y = py + (self.tank_length / 2) * math.sin(rot) + (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c2x = px + (self.tank_length / 2) * math.cos(rot) - (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c2y = py + (self.tank_length / 2) * math.sin(rot) - (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c3x = px - (self.tank_length / 2) * math.cos(rot) + (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c3y = py - (self.tank_length / 2) * math.sin(rot) + (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c4x = px - (self.tank_length / 2) * math.cos(rot) - (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c4y = py - (self.tank_length / 2) * math.sin(rot) - (self.tank_width / 2) * math.sin(rot + math.pi / 2)

        plt.plot(c1x, c1y, 'ro')
        plt.plot(c2x, c2y, 'ro')
        plt.plot(c3x, c3y, 'ro')
        plt.plot(c4x, c4y, 'ro')

        # plt.gca().add_patch(rect)
        # plt.plot([x, x + 0.5 * math.cos(direction)], [y, y + 0.5 * math.sin(direction)], color)

    def draw_tank_pygame(self, x, y, direction, color):

        # draw a rectangle at x, y with the direction
        self.draw_line([x, y], [x + self.tank_length * math.cos(direction), y + self.tank_length * math.sin(direction)],
                       color)

        px = x
        py = y
        rot = direction

        c1x = px + (self.tank_length / 2) * math.cos(rot) + (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c1y = py + (self.tank_length / 2) * math.sin(rot) + (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c2x = px + (self.tank_length / 2) * math.cos(rot) - (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c2y = py + (self.tank_length / 2) * math.sin(rot) - (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c3x = px - (self.tank_length / 2) * math.cos(rot) + (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c3y = py - (self.tank_length / 2) * math.sin(rot) + (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c4x = px - (self.tank_length / 2) * math.cos(rot) - (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c4y = py - (self.tank_length / 2) * math.sin(rot) - (self.tank_width / 2) * math.sin(rot + math.pi / 2)

        self.draw_circle([c1x, c1y], color)
        self.draw_circle([c2x, c2y], color)
        self.draw_circle([c3x, c3y], color)
        self.draw_circle([c4x, c4y], color)

    def project(self, v1, v2):
        # project v1 onto v2
        return v1[0] * v2[0] + v1[1] * v2[1]

    def sat_test(self, points, obstacles, axis):
        # project all points onto the axis
        projections = [self.project(p, axis) for p in points]
        # project all obstacles onto the axis
        obstacle_projections = [self.project(p, axis) for p in obstacles]
        # check if the projections overlap
        return max(projections) >= min(obstacle_projections) and max(obstacle_projections) >= min(projections)

    def normalize(self, v):
        n = sqrt(v[0] * v[0] + v[1] * v[1])
        return [v[0] / n, v[1] / n]

    def perp_axis_from_points(self, p1, p2):
        # get the perpendicular axis to the line defined by p1 and p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return self.normalize([-dy, dx])

    @functools.lru_cache(maxsize=128)
    def get_player_corners(self, px, py, rot):
        c1x = px + (self.tank_length / 2) * math.cos(rot) + (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c1y = py + (self.tank_length / 2) * math.sin(rot) + (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c2x = px + (self.tank_length / 2) * math.cos(rot) - (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c2y = py + (self.tank_length / 2) * math.sin(rot) - (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c3x = px - (self.tank_length / 2) * math.cos(rot) + (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c3y = py - (self.tank_length / 2) * math.sin(rot) + (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c4x = px - (self.tank_length / 2) * math.cos(rot) - (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c4y = py - (self.tank_length / 2) * math.sin(rot) - (self.tank_width / 2) * math.sin(rot + math.pi / 2)

        c1 = [c1x, c1y]
        c2 = [c2x, c2y]
        c3 = [c3x, c3y]
        c4 = [c4x, c4y]

        return c1, c2, c3, c4

    def player_segment_collision(self, px, py, rot, x1, y1, x2, y2):
        # first, we find the 4 corners of the tank
        c1, c2, c3, c4 = self.get_player_corners(px, py, rot)

        # we can use the separating axis theorem to check if the player segment collides with the wall segment
        # lets first project onto the normal of the wall segment

        if not self.sat_test([[x1, y1], [x2, y2]], [c1, c2, c3, c4], [1, 0]):
            return False

        if not self.sat_test([[x1, y1], [x2, y2]], [c1, c2, c3, c4], [0, 1]):
            return False

        # now we project onto the normal of the player segment
        axis = [math.cos(rot + math.pi / 2), math.sin(rot + math.pi / 2)]
        if not self.sat_test([[x1, y1], [x2, y2]], [c1, c2, c3, c4], axis):
            return False

        # now we project onto the player segment
        axis = [math.cos(rot), math.sin(rot)]
        if not self.sat_test([c1, c2, c3, c4], [[x1, y1], [x2, y2]], axis):
            return False

        return True

    def player_collision(self, px, py, rot):
        # we check if the player collides with any wall
        for i in range(self.size_x + 1):
            for j in range(self.size_y + 1):
                if max(abs(px - i), abs(py - j)) > 1:
                    continue
                if self.horizontal_walls[i][j]:
                    if self.player_segment_collision(px, py, rot, i, j, i + 1, j):
                        return True
                if self.vertical_walls[i][j]:
                    if self.player_segment_collision(px, py, rot, i, j, i, j + 1):
                        return True
        return False

    def segment_collision(self, s1, s2):
        s1p1 = [s1[0], s1[1]]
        s1p2 = [s1[2], s1[3]]
        s2p1 = [s2[0], s2[1]]
        s2p2 = [s2[2], s2[3]]
        ax1 = self.normalize([s1[2] - s1[0], s1[3] - s1[1]])
        ax2 = self.normalize([s2[2] - s2[0], s2[3] - s2[1]])
        ax3 = self.perp_axis_from_points(s1p1, s1p2)
        ax4 = self.perp_axis_from_points(s2p1, s2p2)
        if not self.sat_test([s1p1, s1p2], [s2p1, s2p2], ax1):
            return False
        if not self.sat_test([s1p1, s1p2], [s2p1, s2p2], ax2):
            return False
        if not self.sat_test([s1p1, s1p2], [s2p1, s2p2], ax3):
            return False
        if not self.sat_test([s1p1, s1p2], [s2p1, s2p2], ax4):
            return False
        return True

    def ball_step(self, ball, t=0.0):
        eps = 0.01
        if t >= 1:
            return ball.x, ball.y, ball.vx, ball.vy
        next_x = ball.x + (1 - t) * ball.vx
        next_y = ball.y + (1 - t) * ball.vy
        current_x = ball.x
        current_y = ball.y
        segment = [current_x, current_y, next_x, next_y]
        collisions_h = []
        collisions_v = []
        for i in range(self.size_x + 1):
            for j in range(self.size_y + 1):
                if max(abs(ball.x - i), abs(ball.y - j)) > 2:
                    continue
                if self.horizontal_walls[i][j]:
                    if self.segment_collision(segment, [i - eps, j, i + 1 + eps, j]):
                        collision_time = (j - ball.y) / ball.vy
                        collisions_h.append((collision_time, i, j))
                if self.vertical_walls[i][j]:
                    if self.segment_collision(segment, [i, j - eps, i, j + 1 + eps]):
                        collision_time = (i - ball.x) / ball.vx
                        collisions_v.append((collision_time, i, j))
        collisions_h = sorted(collisions_h, key=lambda x: x[0])
        collisions_v = sorted(collisions_v, key=lambda x: x[0])
        if len(collisions_h) == 0 and len(collisions_v) == 0:
            return next_x, next_y, ball.vx, ball.vy
        min_t_h = 10000000 if len(collisions_h) == 0 else collisions_h[0][0]
        min_t_v = 10000000 if len(collisions_v) == 0 else collisions_v[0][0]
        if min(min_t_h, min_t_v) + t > 1 + eps:
            return next_x, next_y, ball.vx, ball.vy
        if min_t_h < min_t_v:
            # we step to the point of collision, and then we invert the y velocity, and we step again
            collision_x = ball.x + ball.vx * (min_t_h)
            collision_y = ball.y + ball.vy * (min_t_h)
            new_vy = -ball.vy
            new_vx = ball.vx
            return self.ball_step(Ball(collision_x, collision_y + eps * new_vy, new_vx, new_vy, ball.life), t + min_t_h)
        else:
            collision_x = ball.x + ball.vx * (min_t_v)
            collision_y = ball.y + ball.vy * (min_t_v)
            new_vx = -ball.vx
            new_vy = ball.vy
            return self.ball_step(Ball(collision_x + eps * new_vx, collision_y, new_vx, new_vy, ball.life), t + min_t_v)

    def player_player_collision(self, p1x, p1y, p1rot, p2x, p2y, p2rot):
        c1, c2, c3, c4 = self.get_player_corners(p1x, p1y, p1rot)
        c5, c6, c7, c8 = self.get_player_corners(p2x, p2y, p2rot)

        if not self.sat_test([c1, c2, c3, c4], [c5, c6, c7, c8], [1, 0]):
            return False
        if not self.sat_test([c1, c2, c3, c4], [c5, c6, c7, c8], [0, 1]):
            return False
        if not self.sat_test([c1, c2, c3, c4], [c5, c6, c7, c8], [math.cos(p1rot), math.sin(p1rot)]):
            return False
        if not self.sat_test([c1, c2, c3, c4], [c5, c6, c7, c8], [math.cos(p2rot), math.sin(p2rot)]):
            return False
        if not self.sat_test([c1, c2, c3, c4], [c5, c6, c7, c8],
                             [math.cos(p1rot + math.pi / 2), math.sin(p1rot + math.pi / 2)]):
            return False
        if not self.sat_test([c1, c2, c3, c4], [c5, c6, c7, c8],
                             [math.cos(p2rot + math.pi / 2), math.sin(p2rot + math.pi / 2)]):
            return False
        return True

    def ball_player_collision(self, ball, px, py, rot):
        c1, c2, c3, c4 = self.get_player_corners(px, py, rot)
        if not self.sat_test([c1, c2, c3, c4], [[ball.x, ball.y]], [1, 0]):
            return False
        if not self.sat_test([c1, c2, c3, c4], [[ball.x, ball.y]], [0, 1]):
            return False
        if not self.sat_test([c1, c2, c3, c4], [[ball.x, ball.y]], [math.cos(rot), math.sin(rot)]):
            return False
        if not self.sat_test([c1, c2, c3, c4], [[ball.x, ball.y]],
                             [math.cos(rot + math.pi / 2), math.sin(rot + math.pi / 2)]):
            return False
        return True

    def action_to_onehot(self, action):
        # we return a onehot encoding of the action
        # directions cannot oppose each other, so we have -> 3 options there
        # we can also shoot or not shoot
        # we can also accelerate or break or neither -> 3 options there
        acc_idx = {
            (False, False): 0,
            (True, False): 1,
            (False, True): 2,
            (True, True): 0
        }
        direction_idx = {
            (False, False): 0,
            (True, False): 1,
            (False, True): 2,
            (True, True): 0
        }
        idx = acc_idx[action[0], action[1]] + 3 * direction_idx[action[2], action[3]] + 9 * action[4]
        return idx

    def onehot_to_action(self, idx):
        # if array, get argmax
        if type(idx) in (list, np.ndarray, tuple):
            idx = np.argmax(np.array(np.array(idx)))
        # we return the action corresponding to the onehot encoding
        acc_idx = idx % 3
        idx = idx // 3
        direction_idx = idx % 3
        idx = idx // 3
        shoot = idx
        acc_array = [[False, False], [True, False], [False, True]]
        direction_array = [[False, False], [True, False], [False, True]]
        return acc_array[acc_idx] + direction_array[direction_idx] + [shoot == 1]

    def set_onehot(self, is_onehot=False):
        if is_onehot:
            self.action_transform = self.onehot_to_action
            self.action_untransform = self.action_to_onehot
            self.action_spaces["0"] = Discrete(18)
            self.action_spaces["1"] = Discrete(18)
            self.action_shape = gymnasium.spaces.flatdim(self.action_spaces["0"])
            self.mask = np.ones(18, dtype=bool)
        else:
            self.action_transform = lambda x: x
            self.action_untransform = lambda x: x
            self.action_spaces["0"] = MultiBinary(5)
            self.action_spaces["1"] = MultiBinary(5)
            self.action_shape = gymnasium.spaces.flatdim(self.action_spaces["0"])
            self.mask = np.ones(5, dtype=bool)

    def step(self, actions):
        acc_time = 15

        # we get the actions from the agents
        p1_action = self.action_transform(actions["0"])
        p2_action = self.action_transform(actions["1"])

        p1_original_x = self.p1_x
        p1_original_y = self.p1_y
        p1_original_direction = self.p1_direction

        p2_original_x = self.p2_x
        p2_original_y = self.p2_y
        p2_original_direction = self.p2_direction

        # we apply the actions to the players
        if p1_action[0]:
            self.p1_v += self.tank_speed / acc_time
        if p1_action[1]:
            self.p1_v -= self.tank_speed / acc_time
        if p1_action[2]:
            self.p1_direction += 0.1
        if p1_action[3]:
            self.p1_direction -= 0.1
        if self.p1_v > 0:
            self.p1_v -= self.tank_speed / (acc_time * 2)
            self.p1_v = max(self.p1_v, 0)
        if self.p1_v < 0:
            self.p1_v += self.tank_speed / (acc_time * 2)
            self.p1_v = min(self.p1_v, 0)

        self.p1_v = min(self.p1_v, self.tank_speed)
        self.p1_v = max(self.p1_v, -self.tank_speed)
        self.p1_direction = self.p1_direction % (2 * math.pi)

        if p2_action[0]:
            self.p2_v += self.tank_speed / acc_time
        if p2_action[1]:
            self.p2_v -= self.tank_speed / acc_time
        if p2_action[2]:
            self.p2_direction += 0.1
        if p2_action[3]:
            self.p2_direction -= 0.1
        if self.p2_v > 0:
            self.p2_v -= self.tank_speed / (acc_time * 2)
            self.p2_v = max(self.p2_v, 0)
        if self.p2_v < 0:
            self.p2_v += self.tank_speed / (acc_time * 2)
            self.p2_v = min(self.p2_v, 0)

        front = [self.p1_x + 0.5 * math.cos(self.p1_direction) * self.tank_length,
                 self.p1_y + 0.5 * math.sin(self.p1_direction) * self.tank_length]

        if p1_action[4]:
            self.p1_balls[1:-1] = self.p1_balls[0:-2]
            self.p1_balls[0] = Ball(front[0], front[1], self.ball_speed * math.cos(self.p1_direction),
                                    self.ball_speed * math.sin(self.p1_direction), self.ball_lifetime)

        front = [self.p2_x + 0.5 * math.cos(self.p2_direction) * self.tank_length,
                 self.p2_y + 0.5 * math.sin(self.p2_direction) * self.tank_length]

        if p2_action[4]:
            self.p2_balls[1:-1] = self.p2_balls[0:-2]
            self.p2_balls[0] = Ball(front[0], front[1], self.ball_speed * math.cos(self.p2_direction),
                                    self.ball_speed * math.sin(self.p2_direction), self.ball_lifetime)

        for ball in self.p1_balls:
            if ball.life > 0:
                ball.x, ball.y, ball.vx, ball.vy = self.ball_step(ball)
                ball.life -= 1
        for ball in self.p2_balls:
            if ball.life > 0:
                ball.x, ball.y, ball.vx, ball.vy = self.ball_step(ball)
                ball.life -= 1

        self.p2_v = min(self.p2_v, self.tank_speed)
        self.p2_v = max(self.p2_v, -self.tank_speed)
        self.p2_direction = self.p2_direction % (2 * math.pi)
        # we move the players
        self.p1_x += self.p1_v * math.cos(self.p1_direction)
        self.p1_y += self.p1_v * math.sin(self.p1_direction)
        self.p2_x += self.p2_v * math.cos(self.p2_direction)
        self.p2_y += self.p2_v * math.sin(self.p2_direction)

        # we check for collisions with the walls
        if self.player_collision(self.p1_x, self.p1_y, self.p1_direction) or self.player_player_collision(self.p1_x,
                                                                                                          self.p1_y,
                                                                                                          self.p1_direction,
                                                                                                          p2_original_x,
                                                                                                          p2_original_y,
                                                                                                          p2_original_direction):
            self.p1_x = p1_original_x
            self.p1_y = p1_original_y
            self.p1_direction = p1_original_direction
            self.p1_v = 0

        if self.player_collision(self.p2_x, self.p2_y, self.p2_direction) or self.player_player_collision(self.p2_x,
                                                                                                          self.p2_y,
                                                                                                          self.p2_direction,
                                                                                                          p1_original_x,
                                                                                                          p1_original_y,
                                                                                                          p1_original_direction):
            self.p2_x = p2_original_x
            self.p2_y = p2_original_y
            self.p2_direction = p2_original_direction
            self.p2_v = 0

        if self.player_player_collision(self.p1_x, self.p1_y, self.p1_direction, self.p2_x, self.p2_y,
                                        self.p2_direction):
            self.p1_x = p1_original_x
            self.p1_y = p1_original_y
            self.p1_direction = p1_original_direction
            self.p1_v = 0
            self.p2_x = p2_original_x
            self.p2_y = p2_original_y
            self.p2_direction = p2_original_direction
            self.p2_v = 0

        observations = [None, None]
        rewards = [0, 0]
        dones = {"0": False, "1": False}
        truncations = dict()
        infos = {"0": {}, "1": {}}
        fill = lambda x, y: [-1 if i >= len(x) else x[i] for i in range(y)]

        observations = self.get_obs()

        self.remaining_time -= 1

        p1_hit = any([self.ball_player_collision(ball, self.p1_x, self.p1_y, self.p1_direction) for ball in
                      self.p2_balls + self.p1_balls])
        p2_hit = any([self.ball_player_collision(ball, self.p2_x, self.p2_y, self.p2_direction) for ball in
                      self.p1_balls + self.p2_balls])

        if p1_hit and p2_hit:
            rewards = [0, 0]
            dones = {"0": True, "1": True}

        if p1_hit:
            is_sucide = any(
                [self.ball_player_collision(ball, self.p1_x, self.p1_y, self.p1_direction) for ball in self.p1_balls])
            rewards = [-1, 1]
            # if is_sucide:
            # rewards = [-10, 0.3]
            dones = {"0": True, "1": True}

        if p2_hit:
            is_sucide = any(
                [self.ball_player_collision(ball, self.p2_x, self.p2_y, self.p2_direction) for ball in self.p2_balls])
            rewards = [1, -1]
            # if is_sucide:
            # rewards = [0.3, -10]
            dones = {"0": True, "1": True}

        if self.remaining_time <= 0:
            dones = {"0": True, "1": True}
            rewards = [0, 0]

        if (int(self.p1_x), int(self.p1_y)) not in self.seen_cells["0"]:
            self.seen_cells["0"].add((int(self.p1_x), int(self.p1_y)))
            rewards[0] += 0.003

        if (int(self.p2_x), int(self.p2_y)) not in self.seen_cells["1"]:
            self.seen_cells["1"].add((int(self.p2_x), int(self.p2_y)))
            rewards[1] += 0.003

        dones = {"0": dones["0"], "1": dones["1"]}
        rewards = {"0": rewards[0], "1": rewards[1]}
        truncations = {"0": dones["0"], "1": dones["1"]}
        infos = {"0": {}, "1": {}}

        if any(dones.values()):
            self.agents = []

        if self.always_render: self.pygame_render()

        # print("Step: ", self.remaining_time, "agents: ", self.agents, "P1: ", self.p1_x, self.p1_y, "P2: ", self.p2_x, self.p2_y, "Rewards: ", rewards, "Dones: ", dones, "Truncations: ", truncations, "Infos: ", infos)
        # print("Actions: ", actions)
        return observations, rewards, dones, truncations, infos

    def get_obs(self):
        fill = lambda x, y: [-1 if i >= len(x) else x[i] for i in range(y)]

        observations = {"0": {"observation": gymnasium.spaces.utils.flatten(self.observation_spaces["0"],
                                                                            (self.get_conv_obs(0).flatten(),
                                                                             self.horizontal_walls, self.vertical_walls,
                                                                            [self.p1_x, self.p1_y, self.p1_v,
                                                                              self.p1_direction],
                                                                             [self.p2_x, self.p2_y, self.p2_v,
                                                                              self.p2_direction],
                                                                             [fill([ball.x for ball in self.p1_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.y for ball in self.p1_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.vx for ball in self.p1_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.vy for ball in self.p1_balls],
                                                                                   self.max_balls)],
                                                                             [fill([ball.x for ball in self.p2_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.y for ball in self.p2_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.vx for ball in self.p2_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.vy for ball in self.p2_balls],
                                                                                   self.max_balls)],
                                                                             fill([ball.life for ball in self.p1_balls],
                                                                                  self.max_balls),
                                                                             fill([ball.life for ball in self.p2_balls],
                                                                                  self.max_balls),
                                                                             [0 if i >= len(self.p1_balls) else 1 if
                                                                             self.p1_balls[i].life > 0 else 0 for i in
                                                                              range(self.max_balls)],
                                                                             [0 if i >= len(self.p2_balls) else 1 if
                                                                             self.p2_balls[i].life > 0 else 0 for i in
                                                                              range(self.max_balls)],
                                                                             [self.remaining_time])),
                              "action_mask": self.mask,
                              },
                        "1": {"observation": gymnasium.spaces.utils.flatten(self.observation_spaces["1"],
                                                                            (self.get_conv_obs(1).flatten(),
                                                                             self.horizontal_walls, self.vertical_walls,
                                                                            [self.p2_x, self.p2_y, self.p2_v,
                                                                              self.p2_direction],
                                                                             [self.p1_x, self.p1_y, self.p1_v,
                                                                              self.p1_direction],
                                                                             [fill([ball.x for ball in self.p2_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.y for ball in self.p2_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.vx for ball in self.p2_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.vy for ball in self.p2_balls],
                                                                                   self.max_balls)],
                                                                             [fill([ball.x for ball in self.p1_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.y for ball in self.p1_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.vx for ball in self.p1_balls],
                                                                                   self.max_balls),
                                                                              fill([ball.vy for ball in self.p1_balls],
                                                                                   self.max_balls)],
                                                                             fill([ball.life for ball in self.p2_balls],
                                                                                  self.max_balls),
                                                                             fill([ball.life for ball in self.p1_balls],
                                                                                  self.max_balls),
                                                                             [0 if i >= len(self.p2_balls) else 1 if
                                                                             self.p2_balls[i].life > 0 else 0 for i in
                                                                              range(self.max_balls)],
                                                                             [0 if i >= len(self.p1_balls) else 1 if
                                                                             self.p1_balls[i].life > 0 else 0 for i in
                                                                              range(self.max_balls)],
                                                                             [self.remaining_time])),
                              "action_mask": self.mask, }
                        }

        return observations

    def get_conv_obs(self, player_idx):
        # we make an image of the scene, of resolution s_x*3, s_y*3
        # we have one channel for the walls, one for the active player, one for the opponent, one for the balls
        img = np.zeros((self.size_x * 3, self.size_y * 3, 5))
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.horizontal_walls[i][j]:
                    img[i * 3:(i + 1) * 3 + 1, j*3, 0] = 1
                if self.vertical_walls[i][j]:
                    img[i * 3, j * 3:(j + 1) * 3 + 1, 0] = 1

        player_pos = (self.p1_x, self.p1_y) if player_idx == 0 else (self.p2_x, self.p2_y)
        opponent_pos = (self.p2_x, self.p2_y) if player_idx == 0 else (self.p1_x, self.p1_y)
        rounded_pos = (int(player_pos[0] * 3), int(player_pos[1] * 3))
        img[rounded_pos[0], rounded_pos[1], 1] = 1
        rounded_pos = (int(opponent_pos[0] * 3), int(opponent_pos[1] * 3))
        img[rounded_pos[0], rounded_pos[1], 2] = 1
        for ball in self.p1_balls:
            if ball.life > 0:
                rounded_pos = (int(ball.x * 3), int(ball.y * 3))
                img[rounded_pos[0], rounded_pos[1], 3] += 1
        for ball in self.p2_balls:
            if ball.life > 0:
                rounded_pos = (int(ball.x * 3), int(ball.y * 3))
                img[rounded_pos[0], rounded_pos[1], 4] += 1

        return img

    def pyplot_show_player_img(self, player_idx):
        img = self.get_conv_obs(player_idx)
        # convert each channel of the image to a 0-255 scale
        imgs = [(np.rot90(img[:,:,i]) * 255).astype(np.uint8) for i in range(5)]
        # concatenate the images
        img = np.concatenate(imgs, axis=1)
        # show the image using pyplot
        plt.imshow(img)





    def render(self, framerate=120.0):
        self.pygame_render()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
