import numpy as np
from torch.distributions.constraints import positive


class Environment:
    def __init__(self, map_abstract, playerX_pos, playerY_pos, goalX_pos, goalY_pos, negative_reward=-1, neutral_reward=0, goal_reward=1000):
        self.env_map = map_abstract
        self.cols, self.rows = self.env_map.shape
        self.negative_reward = negative_reward
        self.neutral_reward = neutral_reward
        self.goal_reward = goal_reward
        if not self.validate_start(playerX_pos, playerY_pos):
            raise ValueError("Player must be set in valid position")
        else:
            self.env_map[playerY_pos, playerX_pos] = 2
        if not self.validate_start(goalX_pos, goalY_pos):
            raise ValueError("Goal must be set in valid position")
        else:
            self.env_map[goalY_pos, goalX_pos] = 5
        if (playerX_pos == goalX_pos) and (playerY_pos == goalY_pos):
            self.env_map[goalY_pos, goalX_pos] = 7
        self.original_map = self.env_map.copy()

    def validate_start(self, X_pos, Y_pos):
        if (Y_pos < 0) or (Y_pos > (self.rows - 1)):
            return False
        if (X_pos < 0) or (X_pos > (self.cols - 1)):
            return False
        if self.env_map[Y_pos, X_pos] == 1:
            return False
        else:
            return True

    def where_am_i(self, state):
        location = np.argwhere((state == 2) | (state == 7))
        return location[0]

    def interaction(self, state, action):
        reward = 0
        new_state = state
        agent_location = self.where_am_i(state)
        agent_row = agent_location[0]
        agent_col = agent_location[1]
        match action:
            case "left":
                requestX_pos = agent_col - 1
                requestY_pos = agent_row
            case "right":
                requestX_pos = agent_col + 1
                requestY_pos = agent_row
            case "up":
                requestX_pos = agent_col
                requestY_pos = agent_row - 1
            case "down":
                requestX_pos = agent_col
                requestY_pos = agent_row + 1
            case _:
                requestX_pos = agent_col
                requestY_pos = agent_row

        if not self.validate_move(requestX_pos, requestY_pos):
            reward = self.negative_reward
        else:
            pos_info = state[requestY_pos, requestX_pos]
            new_state[agent_row, agent_col] = 0
            if pos_info == 5:
                new_state[requestY_pos, requestX_pos] = 7
                reward = self.goal_reward
            else:
                new_state[requestY_pos, requestX_pos] = 2
                reward = self.neutral_reward

        self.env_map = new_state
        return self.env_map, reward

    def validate_move(self, requestX_pos, requestY_pos):
        if (requestY_pos < 0) or (requestY_pos > (self.rows - 1)):
            return False
        if (requestX_pos < 0) or (requestX_pos > (self.cols - 1)):
            return False
        if self.env_map[requestY_pos, requestX_pos] == 1:
            return False
        else:
            return True

    def env_print(self):
        #print(f"# rows: {self.rows}")
        #print(f"# cols: {self.cols}")
        print(self.env_map)

    def reset(self):
        self.env_map = self.original_map.copy()
        return self.env_map.copy()