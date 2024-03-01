import math

from pettingzoo import ParallelEnv

import functools
import random

from copy import copy
from math import sqrt

import matplotlib.pyplot as plt

from gymnasium.spaces import Discrete, MultiDiscrete, Box, Tuple, MultiBinary


class Ball:
    def __init__(self, x=-100.0, y=-100.0, vx=-100.0, vy=-100.0, life=-1):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life

    def return_state(self):
        return self.x, self.y, self.vx, self.vy


class TankTrouble(ParallelEnv):
    metadata = {
        "name": "tanktrouble_v0",
    }

    def __init__(self, s_x=8, s_y=5):
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
        self.possible_agents = [0, 1]
        self.ball_speed = self.tank_speed*2
        self.max_speed = 5.0
        self.p_wall = 0.4
        self.ball_lifetime = 200

        self.min_start_dist = max(self.size_x, self.size_y) / 1.33

        self.p1_v = 0.0
        self.p2_v = 0.0

        self.agent_name_mapping = {0: "p1", 1: "p2"}
        self.max_balls = 5
        self.p1_balls = [Ball() for _ in range(self.max_balls)]
        self.p2_balls = [Ball() for _ in range(self.max_balls)]
        self.horizontal_walls = [[_ for _ in range(self.size_y + 1)] for _ in range(self.size_x)]
        self.vertical_walls = [[_ for _ in range(self.size_y)] for _ in range(self.size_x + 1)]
        self.action_spaces = [None, None]
        self.observation_spaces = [None, None]
        self.action_spaces[0] = MultiBinary(5)
        self.action_spaces[1] = MultiBinary(5)
        self.observation_spaces[0] = Tuple(
            [MultiBinary([self.size_x + 1, self.size_y+1]), MultiBinary([self.size_x+1, self.size_y + 1]),
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # self state
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # other state
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(4, self.max_balls)),  # own balls
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(5, self.max_balls)),  # other balls
             Box(low=0, high=self.remaining_time, shape=(1, self.max_balls)), # own balls remaining life
             Box(low=0, high=self.remaining_time, shape=(1, self.max_balls)), # other balls remaining life
             MultiBinary([self.max_balls]),  # own balls valid
             MultiBinary([self.max_balls]),  # other balls valid
             Box(low=0, high=self.remaining_time, shape=(1,))])
        self.observation_spaces[1] = Tuple(
            [MultiBinary([self.size_x + 1, self.size_y+1]), MultiBinary([self.size_x+1, self.size_y + 1]),
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # self state
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # other state
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(4, self.max_balls)),  # own balls
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(5, self.max_balls)),  # other balls
             Box(low=0, high=self.remaining_time, shape=(1, self.max_balls)), # own balls remaining life
             Box(low=0, high=self.remaining_time, shape=(1, self.max_balls)), # other balls remaining life
             MultiBinary([self.max_balls]),  # own balls valid
             MultiBinary([self.max_balls]),  # other balls valid
             Box(low=0, high=self.remaining_time, shape=(1,))])

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
        self.agents = copy(self.possible_agents)
        self.size_x = 8
        self.size_y = 5

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
        self.remaining_time = 100
        self.p1_balls = [Ball() for _ in range(self.max_balls)]
        self.p2_balls = [Ball() for _ in range(self.max_balls)]

        while True:
            self.horizontal_walls = [[random.random() - (1 if idx in [0, self.size_y] else 0) < self.p_wall for idx in
                                      range(self.size_y + 1)] for _ in
                                     range(self.size_x+1)]  # outside walls are always there
            self.vertical_walls = [
                [random.random() - (1 if idx in [0, self.size_x] else 0) < self.p_wall for _ in range(self.size_y+1)] for
                idx in range(self.size_x + 1)]  # outside walls are always there
            # check if there is a path from p1 to p2
            if self.is_path(self.p1_x, self.p1_y, self.p2_x, self.p2_y):
                break

        print("reset")
        pass

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
        axis = [math.cos(rot + math.pi/2), math.sin(rot + math.pi/2)]
        if not self.sat_test([[x1, y1], [x2, y2]], [c1, c2, c3, c4], axis):
            return False

        # now we project onto the player segment
        axis = [math.cos(rot), math.sin(rot)]
        if not self.sat_test([c1, c2, c3, c4], [[x1, y1], [x2, y2]], axis):
            return False

        return True

    def player_collision(self, px, py, rot):
        # we check if the player collides with any wall
        for i in range(self.size_x+1):
            for j in range(self.size_y+1):
                if self.horizontal_walls[i][j]:
                    if self.player_segment_collision(px, py, rot, i, j, i + 1, j):
                        print("collision", i, j, "vertical")
                        return True
                if self.vertical_walls[i][j]:
                    if self.player_segment_collision(px, py, rot, i, j, i, j + 1):
                        print("collision", i, j, "vertical")
                        perp = self.perp_axis_from_points([i, j], [i, j + 1])
                        projected_player = [self.project([px, py], perp)]
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



    def ball_step(self, ball, t=0):
        eps = 0.00000001
        if t >= 1:
            return ball.x, ball.y, ball.vx, ball.vy
        next_x = ball.x + ball.vx
        next_y = ball.y + ball.vy
        current_x = ball.x + ball.vx * t
        current_y = ball.y + ball.vy * t
        segment = [current_x, current_y, next_x, next_y]
        collisions_h = []
        collisions_v = []
        for i in range(self.size_x+1):
            for j in range(self.size_y+1):
                if self.horizontal_walls[i][j]:
                    if self.segment_collision(segment, [i, j, i + 1, j]):
                        collision_time = (j - ball.y) / ball.vy
                        collisions_h.append((collision_time, i, j))
                if self.vertical_walls[i][j]:
                    if self.segment_collision(segment, [i, j, i, j + 1]):
                        collision_time = (i - ball.x) / ball.vx
                        collisions_v.append((collision_time, i, j))
        collisions_h = sorted(collisions_h, key=lambda x: x[0])
        collisions_v = sorted(collisions_v, key=lambda x: x[0])
        if len(collisions_h) == 0 and len(collisions_v) == 0:
            return next_x, next_y, ball.vx, ball.vy
        min_t_h = 10000000 if len(collisions_h) == 0 else collisions_h[0][0]
        min_t_v = 10000000 if len(collisions_v) == 0 else collisions_v[0][0]
        if min(min_t_h, min_t_v) + t > 1:
            return next_x, next_y, ball.vx, ball.vy
        if min_t_h < min_t_v:
            # we step to the point of collision, and then we invert the y velocity, and we step again
            collision_x = ball.x + ball.vx * min_t_h
            collision_y = collisions_h[0][2]
            new_vy = -ball.vy
            new_vx = ball.vx
            return self.ball_step(Ball(collision_x, collision_y+eps, new_vx, new_vy, ball.life), t + min_t_h)
        else:
            collision_x = collisions_v[0][1]
            collision_y = ball.y + ball.vy * min_t_v
            new_vx = -ball.vx
            new_vy = ball.vy
            return self.ball_step(Ball(collision_x + eps, collision_y, new_vx, new_vy, ball.life), t + min_t_v)

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
        if not self.sat_test([c1, c2, c3, c4], [c5, c6, c7, c8], [math.cos(p1rot + math.pi/2), math.sin(p1rot + math.pi/2)]):
            return False
        if not self.sat_test([c1, c2, c3, c4], [c5, c6, c7, c8], [math.cos(p2rot + math.pi/2), math.sin(p2rot + math.pi/2)]):
            return False
        return True




    def step(self, actions):
        acc_time = 15

        # we get the actions from the agents
        p1_action = actions[0]
        p2_action = actions[1]

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

        front = [self.p1_x + 0.5 * math.cos(self.p1_direction) * self.tank_length, self.p1_y + 0.5 * math.sin(self.p1_direction) * self.tank_length]

        if p1_action[4]:
            self.p1_balls[1:-1] = self.p1_balls[0:-2]
            self.p1_balls[0] = Ball(front[0], front[1], self.ball_speed * math.cos(self.p1_direction), self.ball_speed * math.sin(self.p1_direction), self.ball_lifetime)
        if p2_action[4]:
            self.p2_balls[1:-1] = self.p2_balls[0:-2]
            self.p2_balls[0] = Ball(front[0], front[1], self.ball_speed * math.cos(self.p2_direction), self.ball_speed * math.sin(self.p2_direction), self.ball_lifetime)

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
        if self.player_collision(self.p1_x, self.p1_y, self.p1_direction) or self.player_player_collision(self.p1_x, self.p1_y, self.p1_direction, p2_original_x, p2_original_y, p2_original_direction):
            self.p1_x = p1_original_x
            self.p1_y = p1_original_y
            self.p1_direction = p1_original_direction
            self.p1_v = 0

        if self.player_collision(self.p2_x, self.p2_y, self.p2_direction) or self.player_player_collision(self.p2_x, self.p2_y, self.p2_direction, p1_original_x, p1_original_y, p1_original_direction):
            self.p2_x = p2_original_x
            self.p2_y = p2_original_y
            self.p2_direction = p2_original_direction
            self.p2_v = 0

        if self.player_player_collision(self.p1_x, self.p1_y, self.p1_direction, self.p2_x, self.p2_y, self.p2_direction):
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
        dones = [False, False]
        truncations = [False, False]
        infos = [None, None]

        self.remaining_time -= 1

        if self.remaining_time <= 0:
            dones = [True, True]

        return observations, rewards, dones, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
