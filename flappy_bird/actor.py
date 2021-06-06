import pygame
from pygame import K_SPACE


class Actor(object):

    def is_up(self, t_h: int, b_h: int, v_speed: int, to_pipe: int):
        pass

    def finish(self):
        pass


class KeyboardActor(Actor):

    def is_up(self, t_h: int, b_h: int, v_speed: int, to_pipe: int):
        return pygame.key.get_pressed()[K_SPACE]

    def finish(self):
        print("Keyboard actor finished")