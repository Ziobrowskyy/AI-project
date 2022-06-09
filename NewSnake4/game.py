import pygame
import random
from enum import Enum, IntEnum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('../arial.ttf', 25)


class Direction(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class Tile(IntEnum):
    EMPTY = 0
    SNAKE = 2
    FOOD = 3
    WALL = 4


class MoveResult(Enum):
    LIVE = 1,
    ATE_FOOD = 2,
    DIE = 3


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20
SPEED = 10

DISPLAY = True
MULTI_FOOD = False
PLACE_WALLS = True


class SnakeGame:
    def __init__(self, w, h):
        self.head = None
        self.snake = None
        self.direction = None
        self.food = None
        self.score = None
        self.frame_iteration = None
        self.board = None

        self.tile_size = 20

        self.w = w
        self.h = h

        # init display
        if DISPLAY:
            self.display = pygame.display.set_mode((w * self.tile_size, h * self.tile_size))
            pygame.display.set_caption('Wonsz')
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # reset board
        self._clear_board()

        self.direction = Direction.RIGHT

        self.head = Point(self.w // 2, self.h // 2)

        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.x),
            Point(self.head.x - 3, self.head.x),
            Point(self.head.x - 4, self.head.x),
        ]

        for p_x, p_y in self.snake:
            self._set_board_value(p_x, p_y, Tile.SNAKE)

        self.score = 0
        self.food = None
        if PLACE_WALLS:
            self._place_walls()
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, self.w - 1)
        y = random.randint(0, self.h - 1)

        if self.board[y, x] != Tile.EMPTY:
            return self._place_food()

        self._set_board_value(x, y, Tile.FOOD)
        self.food = Point(x, y)

    def _place_walls(self):
        x = random.randint(0, self.w // 3)
        for i in range(0, self.h // 2 - 1):
            self._set_board_value(x, i, Tile.WALL)
            # self.board[i][x] = Tile.WALL

        x = (x + self.w // 2) % self.w
        for i in range(0, self.h // 2 - 1):
            self._set_board_value(x, self.h - i - 1, Tile.WALL)
            # self.board[self.w - i - 1][x] = Tile.WALL

    def _clear_board(self):
        self.board = np.full((self.h, self.w), Tile.EMPTY)

    def _set_board_value(self, x: int, y: int, value: Tile):
        self.board[y][x] = value

    def _get_board_value(self, x: int, y: int) -> Tile:
        return self.board[y, x]

    def _action_to_dir(self, action) -> Direction:
        direction = np.argmax(action[0])
        return Direction(direction)

    def play_step(self, action) -> bool:
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # get direction from model action
        direction = self._action_to_dir(action)
        # move snake
        move_result = self._move_snake(direction)

        # check for game taking to long
        if self.frame_iteration > self.w * len(self.snake):
            move_result = MoveResult.DIE

        move_reward = 0
        match move_result:
            case MoveResult.ATE_FOOD:
                self.score += 1
                self._place_food()
            case MoveResult.DIE:
                self.score -= 0.5
                return False

        # 5. update ui and clock
        if DISPLAY:
            self._update_ui()

        if MULTI_FOOD and self.frame_iteration % 30 == 0:
            self._place_food()

        # self.clock.tick(SPEED)
        # 6. return game over and score
        # self.score += 0.01
        return True

    def _move_snake(self, direction: Direction) -> MoveResult:
        new_head = None
        match direction:
            case Direction.RIGHT:
                if self.direction is Direction.LEFT:
                    return MoveResult.DIE
                new_head = Point(self.head.x + 1, self.head.y)
            case Direction.LEFT:
                if self.direction is Direction.RIGHT:
                    return MoveResult.DIE
                new_head = Point(self.head.x - 1, self.head.y)
            case Direction.UP:
                if self.direction is Direction.DOWN:
                    return MoveResult.DIE
                new_head = Point(self.head.x, self.head.y + 1)
            case Direction.DOWN:
                if self.direction is Direction.UP:
                    return MoveResult.DIE
                new_head = Point(self.head.x, self.head.y - 1)

        self.direction = direction

        if not 0 <= new_head.x < self.w or not 0 <= new_head.y < self.h:
            return MoveResult.DIE

        self.head = new_head
        # self.snake.insert(0, new_head)
        tile_at_head = self._get_board_value(self.head.x, self.head.y)

        match tile_at_head:
            case Tile.EMPTY | Tile.FOOD:
                self.snake.insert(0, self.head)
                self._set_board_value(self.head.x, self.head.y, Tile.SNAKE)

                if tile_at_head == Tile.FOOD:
                    return MoveResult.ATE_FOOD

                last_tail = self.snake.pop()
                self._set_board_value(last_tail.x, last_tail.y, Tile.EMPTY)
                return MoveResult.LIVE

            case Tile.WALL | Tile.SNAKE:
                return MoveResult.DIE

    def get_model_data(self):
        wall_0, body_0, food_0 = self._get_dist(1, 0)  # right
        wall_1, body_1, food_1 = self._get_dist(1, 1)  # top right
        wall_2, body_2, food_2 = self._get_dist(0, 1)  # top
        wall_3, body_3, food_3 = self._get_dist(-1, 1)  # top left
        wall_4, body_4, food_4 = self._get_dist(-1, 0)  # left
        wall_5, body_5, food_5 = self._get_dist(-1, -1)  # bottom left
        wall_6, body_6, food_6 = self._get_dist(0, -1)  # bottom
        wall_7, body_7, food_7 = self._get_dist(1, -1)  # bottom right
        return np.array([
            wall_0, body_0, food_0,
            wall_1, body_1, food_1,
            wall_2, body_2, food_2,
            wall_3, body_3, food_3,
            wall_4, body_4, food_4,
            wall_5, body_5, food_5,
            wall_6, body_6, food_6,
            wall_7, body_7, food_7,
        ])

    def _get_dist(self, dx: int, dy: int):
        return np.array([
            self._get_dist_to_wall(dx, dy),
            self._get_dist_to_body(dx, dy),
            self._get_dist_to_food(dx, dy),
        ])

    def _get_dist_to_wall(self, dx: int, dy: int) -> float:
        dist = 0
        x = self.head.x
        y = self.head.y
        while True:
            dist += 1
            if self._get_board_value(x, y) is Tile.WALL:
                break
            x += dx
            y += dy
            if not 0 <= x < self.w or not 0 <= y < self.h:
                break
        return 1 / dist

    def _get_dist_to_body(self, dx: int, dy: int) -> int:
        x = self.head.x
        y = self.head.y
        while True:
            if Point(x, y) in self.snake[1:]:
                return 1
            x += dx
            y += dy
            if not 0 <= x < self.w or not 0 <= y < self.h:
                return 0

    def _get_dist_to_food(self, dx: int, dy: int) -> int:
        x = self.head.x
        y = self.head.y
        while True:
            if self._get_board_value(x, y) is Tile.FOOD:
                return 1
            x += dx
            y += dy
            if not 0 <= x < self.w or not 0 <= y < self.h:
                return 0

    def _update_ui(self):
        self.display.fill(BLACK)

        for y in range(self.h):
            for x in range(self.w):
                p_x = x * BLOCK_SIZE
                p_y = y * BLOCK_SIZE
                match self.board[y][x]:
                    case Tile.EMPTY:
                        pygame.draw.rect(self.display, BLACK, pygame.Rect(p_x, p_y, BLOCK_SIZE, BLOCK_SIZE))
                    case Tile.SNAKE:
                        pygame.draw.rect(self.display, BLUE1, pygame.Rect(p_x, p_y, BLOCK_SIZE, BLOCK_SIZE))
                    case Tile.FOOD:
                        pygame.draw.rect(self.display, RED, pygame.Rect(p_x, p_y, BLOCK_SIZE, BLOCK_SIZE))
                    case Tile.WALL:
                        pygame.draw.rect(self.display, WHITE, pygame.Rect(p_x, p_y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


if __name__ == '__main__':
    game = SnakeGame(20, 20)
    game.direction = Direction.UP
    while True:
        is_alive, step_score, game_score = game.play_step([1, 0, 0])
        if not is_alive:
            break
    print(f"you died, score is: {game_score}")
