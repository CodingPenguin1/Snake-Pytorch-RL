from random import randint


class Game:
    def __init__(self, board_size=(10, 10)):
        self.width, self.height = board_size
        assert self.width >= 4
        assert self.height >= 4

        self.snake = []
        self.direction = 'r'
        self.food = None
        self.iterations = 0
        self.score = 0

        self.reset()

    def reset(self):
        head_coords = (self.width // 2, self.height // 2)
        self.snake = [
            [head_coords[0], head_coords[1]],
            [head_coords[0], head_coords[1] - 1],
            [head_coords[0], head_coords[1] - 2]
        ]
        self.direction = 'r'

        self.iterations = 0
        self.score = 0
        self._place_food()

    def _place_food(self):
        row, col = randint(0, self.height - 1), randint(0, self.width - 1)
        while [row, col] in self.snake:
            row, col = randint(0, self.height - 1), randint(0, self.width - 1)
        self.food = [row, col]

    def step(self, action):
        self.iterations += 1

        ate_food = self._move(action)
        game_over = self.is_collision() or self.iterations > 100 * len(self.snake)

        return self.score, ate_food, game_over

    def is_collision(self, point=None):
        if point is None:
            point = self.snake[0]

        # Out of bounds
        if point[0] < 0 or point[0] >= self.height or point[1] < 0 or point[1] >= self.width:
            return True

        # Hits itself
        return point in self.snake[1:]

    def _move(self, action):
        # [straight, right, left]

        # Compute new direction from action
        clockwise = ['r', 'd', 'l', 'u']
        idx = clockwise.index(self.direction)

        if action[1]:
            new_dir = clockwise[idx]
        elif action[0]:
            new_dir = clockwise[(idx - 1) % 4]
        else:
            new_dir = clockwise[(idx + 1) % 4]
        self.direction = new_dir

        # Compute new head position
        row = self.snake[0][0]
        col = self.snake[0][1]
        if self.direction == 'r':
            col += 1
        elif self.direction == 'l':
            col -= 1
        elif self.direction == 'u':
            row -= 1
        else:
            row += 1
        new_head = [row, col]

        # If new head is food, add it to snake and place new food
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self._place_food()
            return True  # True for eating food
        # Otherwise, move snake forward one step
        else:
            self.snake.pop()
            return False  # False for not eating food

    def __str__(self):
        s = ''
        for row in range(self.height):
            for col in range(self.width):
                point = [row, col]
                cell_value = '.'
                if point == self.food:
                    cell_value = 'F'
                elif point == self.snake[0]:
                    cell_value = 'H'
                elif point in self.snake:
                    cell_value = 'o'
                s += f'{cell_value} '
            s += '\n'
        return s
