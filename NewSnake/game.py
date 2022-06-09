import pygame
import random
from enum import Enum, IntEnum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('../arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Tile(IntEnum):
    EMPTY = 0
    SNAKE_HEAD = 1
    SNAKE_BODY = 2
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


class SnakeGame:
    def __init__(self, w, h):
        self.head = None
        self.snake = None
        self.direction = None
        self.food = None
        self.score = 0
        self.frame_iteration = None
        self.score = None
        self.board = None

        self.tile_size = 20

        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((w * self.tile_size, h * self.tile_size))
        pygame.display.set_caption('Wonsz')
        self.clock = pygame.time.Clock()
        self.reset()

    def _clear_board(self):
        self.board = np.full((self.h, self.w), Tile.EMPTY)

    def _set_board_value(self, x: int, y: int, value: Tile):
        self.board[y][x] = value

    def _get_board_value(self, x: int, y: int) -> Tile:
        return self.board[y, x]

    def get_model_data(self):
        return np.array(self.board / 4, dtype=int)

    def get_model_data_2(self):
        wall_0, body_0, food_0 = (self._get_dist(1, 0) / self.w)  # right
        wall_1, body_1, food_1 = self._get_dist(1, 1)  # top right
        wall_2, body_2, food_2 = self._get_dist(0, 1)  # top
        wall_3, body_3, food_3 = self._get_dist(-1, 1)  # top left
        wall_4, body_4, food_4 = self._get_dist(-1, 0)  # left
        wall_5, body_5, food_5 = self._get_dist(-1, -1)  # bottom left
        wall_6, body_6, food_6 = self._get_dist(0, -1)  # bottom
        wall_7, body_7, food_7 = self._get_dist(1, -1)  # bottom right
        return [
            wall_0, body_0, food_0,
            wall_1, body_1, food_1,
            wall_2, body_2, food_2,
            wall_3, body_3, food_3,
            wall_4, body_4, food_4,
            wall_5, body_5, food_5,
            wall_6, body_6, food_6,
            wall_7, body_7, food_7,
        ]

    def _get_dist(self, dx: int, dy: int):
        dist_to_wall = 0
        x = self.head.x
        y = self.head.y
        while True:
            if self._get_board_value(x, y) is Tile:
                break
            dist_to_wall += 1
            x += dx
            y += dy
            if not 0 <= x < self.w or not 0 <= y < self.h:
                break
        return np.array([
            dist_to_wall,
            self._get_dist_to_tile(dx, dy, Tile.SNAKE_BODY),
            self._get_dist_to_tile(dx, dy, Tile.FOOD),
        ])

    def _get_dist_to_tile(self, dx: int, dy: int, tile: Tile) -> int:
        dist = 0
        x = self.head.x
        y = self.head.y
        while 0 <= x < self.w and 0 <= y < self.h:
            if self._get_board_value(x, y) is tile:
                break
            dist += 1
            x += dx
            y += dy
        return dist

    def reset(self):
        # reset board
        self._clear_board()

        self.direction = Direction.RIGHT

        self.head = Point(self.w // 2, self.h // 2)

        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.x)
        ]

        for p_x, p_y in self.snake:
            self._set_board_value(p_x, p_y, Tile.SNAKE_BODY)
        self._set_board_value(self.head.x, self.head.y, Tile.SNAKE_HEAD)

        self.score = 0
        self.food = None
        self._place_food()
        # self._set_board_value(self.w // 2 + 4, self.h // 2, Tile.FOOD)
        # self.food = Point(self.w // 2 + 4, self.h // 2)
        self.frame_iteration = 0

    def _move_snake(self) -> MoveResult:
        new_head = None
        match self.direction:
            case Direction.RIGHT:
                new_head = Point(self.head.x + 1, self.head.y)
            case Direction.LEFT:
                new_head = Point(self.head.x - 1, self.head.y)
            case Direction.UP:
                new_head = Point(self.head.x, self.head.y + 1)
            case Direction.DOWN:
                new_head = Point(self.head.x, self.head.y - 1)

        if not 0 <= new_head.x < self.w or not 0 <= new_head.y < self.h:
            return MoveResult.DIE

        self.head = new_head
        # self.snake.insert(0, new_head)
        tile_at_head = self._get_board_value(self.head.x, self.head.y)

        match tile_at_head:
            case Tile.EMPTY | Tile.FOOD:
                self.snake.insert(0, self.head)
                self._set_board_value(self.head.x, self.head.y, Tile.SNAKE_HEAD)

                if tile_at_head == Tile.FOOD:
                    return MoveResult.ATE_FOOD

                last_tail = self.snake.pop()
                self._set_board_value(last_tail.x, last_tail.y, Tile.EMPTY)
                return MoveResult.LIVE

            case Tile.WALL | Tile.SNAKE_BODY:
                return MoveResult.DIE

    def _place_food(self):
        x = random.randint(0, self.w - 1)
        y = random.randint(0, self.h - 1)

        if self.board[y, x] != Tile.EMPTY:
            self._place_food()

        self._set_board_value(x, y, Tile.FOOD)
        self.food = Point(x, y)

    def _place_walls(self):
        y = random.randint(0, self.h - 1)
        for i in range(0, self.w // 2):
            self.board[y][i] = Tile.WALL

        x = random.randint(0, self.w - 1)
        for i in range(0, self.h // 2):
            self.board[i][x] = Tile.WALL

    def _action_to_dir(self, action) -> Direction:
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            pass
        elif np.array_equal(action, [0, 1, 0]):
            idx = (idx + 1) % 4
        elif np.array_equal(action, [0, 0, 1]):
            idx = (idx + 4 - 1) % 4
        new_dir = clockwise[idx]
        return new_dir

    def play_step(self, action) -> (bool, int, int):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # get direction from model action
        self.direction = self._action_to_dir(action)
        # move snake
        move_result = self._move_snake()

        # check for game taking to long
        if self.frame_iteration > 30 * len(self.snake):
            self.score -= 0.1
            move_result = MoveResult.DIE

        move_reward = 0
        match move_result:
            case MoveResult.ATE_FOOD:
                self.score = self.score + 0.1
                move_reward = 10
                self._place_food()
            case MoveResult.DIE:
                self.score = self.score
                return False, 0, self._get_score()

        # 5. update ui and clock
        self._update_ui()
        # 6. return game over and score
        return True, move_reward, self._get_score()

    def _get_score(self):
        return self.score
        # return (len(self.snake) - 3) / 10

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        print("head", self.head)
        if 0 <= pt.x > self.w or 0 <= pt.y < self.h:
            return True
        # if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
        #     return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for y in range(self.h):
            for x in range(self.w):
                p_x = x * BLOCK_SIZE
                p_y = y * BLOCK_SIZE
                match self.board[y][x]:
                    case Tile.EMPTY:
                        pygame.draw.rect(self.display, BLACK, pygame.Rect(p_x, p_y, BLOCK_SIZE, BLOCK_SIZE))
                    case Tile.SNAKE_HEAD:
                        pygame.draw.rect(self.display, BLUE1, pygame.Rect(p_x, p_y, BLOCK_SIZE, BLOCK_SIZE))
                    case Tile.SNAKE_BODY:
                        pygame.draw.rect(self.display, BLUE2, pygame.Rect(p_x, p_y, BLOCK_SIZE, BLOCK_SIZE))
                    case Tile.FOOD:
                        pygame.draw.rect(self.display, RED, pygame.Rect(p_x, p_y, BLOCK_SIZE, BLOCK_SIZE))
                    case Tile.WALL:
                        pygame.draw.rect(self.display, WHITE, pygame.Rect(p_x, p_y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self._get_score()), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


if __name__ == '__main__':
    game = SnakeGame(20, 20)
    game.direction = Direction.UP
    game_score = 0
    while True:
        is_alive, step_score, game_score = game.play_step([1, 0, 0])
        if not is_alive:
            break
    print(f"you died, score is: {game_score}")
