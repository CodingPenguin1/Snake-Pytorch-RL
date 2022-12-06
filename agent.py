import contextlib
import random
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from game import Game
from model import Linear_QNet, QTrainer


BOARD_SIZE = (10, 10)


class Agent:
    def __init__(self, board_size, max_memory=100_000, batch_size=1000, lr=0.001):
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.lr = lr

        self.n_games = 0
        self. epsilon = 0  # randomness
        self. gamma = 0.9  # discount rate
        self.memory = deque(maxlen=self.max_memory)

        self.model = Linear_QNet(board_size[0] * board_size[1], 256, 3)
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)

        self.state = None  # store the state array persistently so we don't have to keep allocating it

    def get_state(self, game):
        # 0 = empty, 1 = food, 2 = snake, 3 = head, 4 = tail

        if self.state is None:
            self.state = np.zeros((game.height, game.width), dtype=np.uint8)
        else:
            self.state.fill(0)

        self.state[game.food[0], game.food[1]] = 1
        head = game.snake[0]
        with contextlib.suppress(IndexError):
            self.state[head[0], head[1]] = 3
        for block in game.snake[1:-1]:
            self.state[block] = 2
        tail = game.snake[-1]
        self.state[tail[0], tail[1]] = 4

        return self.state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
        else:
            batch = self.memory

        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 100 - self.n_games

        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move


def train(board_size):
    record = 0
    agent = Agent(board_size)
    game = Game(board_size)
    writer = SummaryWriter('runs')

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        score, ate_food, done = game.step(final_move)
        # print(game)
        # print(f'Score: {score}, Ate food: {ate_food}, Done: {done}')
        # input()
        reward = 0
        if ate_food:
            reward = 10
        if done:
            reward = -10
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # print(game)
            print(f'Score: {score}, Iterations: {game.iterations}')
            # input()

            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')

            writer.add_scalar('Score', score, agent.n_games)
            writer.flush()


if __name__ == '__main__':
    train(BOARD_SIZE)
