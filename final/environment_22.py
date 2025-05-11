# Task 2.2 - Building environment

import numpy as np

class Environment:
    """Environment for reinforcement learning. A 2d numpy array.
    Parameters
    ------------
    map_abstract : 2d numpy array
    map should be built from map_abstraction class.
    playerX_pos : int
    player/agent starting x position.
    playerY_pos : int
    player/agent starting y position.
    goalX_pos : int
    goals x position.
    goalY_pos : int
    goals y position.
    negative_reward : int
    penalization value for trying to move into objects or out of the numpy array.
    neutral_reward : int
    reward for moving around the environment.
    goal_reward : int
    reward for finding the goal.
    Attributes
    -----------
    env_map : 2d numpy array
    the environment, shown as a map (2d numpy array).
        Values in the env_map:
         0 : free move location
         1 : obstacle
         2 : agent
         5 : goal
         7 : agent and goal when agent is on the goal position
    cols : int
    width of numpy array.
    rows : int
    height of numpy array.
    original_map : 2d numpy array
    starting map, used when starting a new episode.
    """
    def __init__(self, map_abstract, playerX_pos, playerY_pos, goalX_pos, goalY_pos, negative_reward=-1, neutral_reward=0, goal_reward=1000):
        self.env_map = map_abstract
        self.cols, self.rows = self.env_map.shape
        self.negative_reward = negative_reward
        self.neutral_reward = neutral_reward
        self.goal_reward = goal_reward

        #conditionals to not allow initialization if player or goal position are not correct.
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
        """Validate user requested starting positions, if valid return true, else return false.
        Parameters
        ----------
        X_pos : int
        requested x position.
        Y_pos : int
        requested y position.
        Returns
        -------
        bool
        """
        if (Y_pos < 0) or (Y_pos > (self.rows - 1)):
            return False
        if (X_pos < 0) or (X_pos > (self.cols - 1)):
            return False
        if self.env_map[Y_pos, X_pos] == 1:
            return False
        else:
            return True

    def where_am_i(self, state):
        """Tells where the agent is on the map.
        Parameters
        ----------
        state : 2d numpy array
        current environment.
        Returns
        -------
        array : numpy array
        gives agent position.
        """
        location = np.argwhere((state == 2) | (state == 7))
        return location[0]

    def interaction(self, state, action):
        """Takes in current environment state and requested agent action
        and returns updated state and reward for action.
        Parameters
        ----------
        state : 2d numpy array
        current environment.
        Returns
        -------
        env_map : 2d numpy array
        state of the environment after requested agent action.
        reward : int
        reward for requested action.
        """
        #copy initial state and find agent position.
        new_state = state
        agent_location = self.where_am_i(state)
        agent_row = agent_location[0]
        agent_col = agent_location[1]

        #get requested agent position.
        if action == "left":
            requestX_pos = agent_col - 1
            requestY_pos = agent_row
        elif action == "right":
            requestX_pos = agent_col + 1
            requestY_pos = agent_row
        elif action == "up":
            requestX_pos = agent_col
            requestY_pos = agent_row - 1
        elif action == "down":
            requestX_pos = agent_col
            requestY_pos = agent_row + 1
        else:
            requestX_pos = agent_col
            requestY_pos = agent_row

        #check if move is valid, update agent position, and reward.
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

        #update environment with the latest state.
        self.env_map = new_state

        #return new state and reward.
        return self.env_map, reward

    def validate_move(self, requestX_pos, requestY_pos):
        """Validate agent requested position, if valid return true, else return false.
        Parameters
        ----------
        X_pos : int
        requested x position.
        Y_pos : int
        requested y position.
        Returns
        -------
        bool
        """
        if (requestY_pos < 0) or (requestY_pos > (self.rows - 1)):
            return False
        if (requestX_pos < 0) or (requestX_pos > (self.cols - 1)):
            return False
        if self.env_map[requestY_pos, requestX_pos] == 1:
            return False
        else:
            return True

    def env_print(self):
        """Helper function that prints environment map.
        """
        print(self.env_map)

    def reset(self):
        """Reset environment for next episode.
        Returns
        -------
        env_map : 2d numpy array
        sends the original starting map.
        """
        self.env_map = self.original_map.copy()
        return self.env_map.copy()
