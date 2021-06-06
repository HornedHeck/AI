import os
import random
import sys

import pygame
from pygame.locals import *
from pygame.sprite import AbstractGroup, Group, Sprite

import ai
from flappy_bird.actor import Actor

WHITE = (255, 255, 255)
H_SPEED = -8
UP_V_SPEED = -10
MAX_V_SPEED = 15

SCREEN_H = 600
SCREEN_W = 600
PIPE_GAP = 200
PIPE_DISTANCE = 300

pipe_rand = random.Random(10)


class Pipe(Group):

    def __init__(self) -> None:
        self.upper = Sprite()
        self.upper.image = pygame.transform.flip(pygame.image.load(resource_path("assets/pipe-green.png")), False, True)

        upper_h = pipe_rand.randint(
            SCREEN_H - PIPE_GAP - self.upper.image.get_height(),
            self.upper.image.get_height()
        )

        self.upper.rect = self.upper.image.get_rect().move(
            SCREEN_W + PIPE_DISTANCE,
            upper_h - self.upper.image.get_height()
        )

        self.bottom = Sprite()
        self.bottom.image = pygame.image.load(resource_path("assets/pipe-green.png"))
        self.bottom.rect = self.bottom.image.get_rect().move(
            SCREEN_W + PIPE_DISTANCE,
            upper_h + PIPE_GAP
        )
        super().__init__(self.upper, self.bottom)

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

        for sprite in self.sprites():
            sprite.rect.move_ip(H_SPEED, 0)


class Bird(pygame.sprite.Sprite):
    __v_speed = 0

    def __init__(self, actor: Actor, *groups: AbstractGroup) -> None:
        super().__init__(*groups)
        self.actor = actor
        self.image_up = pygame.image.load(resource_path("assets/bird_up.png"))
        self.image_mid = pygame.image.load(resource_path("assets/bird_mid.png"))
        self.image_down = pygame.image.load(resource_path("assets/bird_down.png"))
        self.image = self.image_mid
        self.rect = self.image.get_rect(center=(100, SCREEN_H / 2))

    def move(self, pipe: Pipe):

        t_h = self.rect.top - pipe.upper.rect.bottom
        b_h = pipe.bottom.rect.top - self.rect.bottom
        to_pipe = pipe.upper.rect.x - self.rect.right
        if self.actor.is_up(t_h, b_h, self.__v_speed, to_pipe):
            self.__v_speed = UP_V_SPEED
        else:
            self.__v_speed = min(self.__v_speed + 1, MAX_V_SPEED)

        self.rect.move_ip(0, self.__v_speed)
        if self.rect.y < 0:
            self.rect.move_ip(0, -self.rect.y)
            self.__v_speed = 0

        if self.__v_speed < -1:
            self.image = self.image_down
        elif self.__v_speed < 1:
            self.image = self.image_mid
        else:
            self.image = self.image_up


class FlappyBirdGame(object):

    def __init__(self) -> None:
        super().__init__()
        self.__DISPLAY__ = pygame.display.set_mode((SCREEN_H, SCREEN_W))
        self.__FPS__ = pygame.time.Clock()

    def start(self, models):

        birds = []

        for i in range(len(models)):
            birds.append(Bird(ai.NeuroActor(models[i])))

        pipes = [Pipe()]

        is_paused = False

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    is_paused = not is_paused

            if is_paused:
                pygame.display.update()
                self.__FPS__.tick(10)
                continue

            self.__DISPLAY__.fill(WHITE)

            for pipe in pipes:
                pipe.update()
                pipe.draw(self.__DISPLAY__)

            if pipes[-1].upper.rect.x < SCREEN_W + 10:
                pipes.append(Pipe())

            if pipes[0].upper.rect.x < -pipes[0].upper.rect.width:
                del pipes[0]

            for bird in birds:
                if pipes[0].upper.rect.left >= bird.rect.right:
                    bird.move(pipes[0])
                else:
                    bird.move(pipes[1])
                self.__DISPLAY__.blit(bird.image, bird.rect)

                if bird.rect.bottom >= SCREEN_H - 1:
                    bird.actor.finish()
                    birds.remove(bird)

                for pipe in pipes:
                    if pygame.sprite.spritecollideany(bird, pipe):
                        bird.actor.finish()
                        birds.remove(bird)

            if len(birds) == 0:
                return

            pygame.display.update()
            self.__FPS__.tick(60)


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def model_to_string(model: ai.Model):
    return f"""Weights:{
    model.fc1.weight.detach().numpy()
    } Bias: {
    model.fc1.bias[0].item()
    } Score: {
    model.fitness
    }"""


def start_learning():
    game = FlappyBirdGame()

    m1 = ai.Model()

    m2 = ai.Model()

    ai.finished__generation = [m1, m2]

    generation = 1
    while True:
        # run game
        game.start(ai.new_generation())

        print(f"Generation {generation}:")
        print(f"1. {model_to_string(ai.finished__generation[-1])}")
        print(f"2. {model_to_string(ai.finished__generation[-2])}")

        # check continuation
        # if generation % 5 == 0 and input('Continue (any/n)?:') == 'n':
        #     pygame.quit()
        #     sys.exit(0)

        generation += 1


def start_showcase():
    game = FlappyBirdGame()

    best_model = ai.Model()
    best_model.fc1.bias[0] = -0.05229160189628601
    best_model.fc1.weight[0, 0] = -0.02542654
    best_model.fc1.weight[0, 1] = -1.8577229
    best_model.fc1.weight[0, 2] = -0.04142456
    best_model.fc1.weight[0, 3] = -0.4075667

    game.start([best_model])

    print(best_model.fitness)


if __name__ == '__main__':
    pygame.init()

    start_learning()
    # init start parents
