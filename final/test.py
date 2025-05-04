from map_abstraction_21 import Map_Abstraction
from environment_22 import Environment
from agent_234 import Agent

ma = Map_Abstraction()
np_img_array = ma.abstract("./maps/map1.bmp", 20, 20)
np_img_array2 = ma.abstract("./maps/map2.bmp", 40, 40)
'''
print("shape:")
print(np_img_array.shape)
print("dtype:")
print(np_img_array.dtype)
'''
ma.show_plot(np_img_array, show_grid=True)
ma.show_plot(np_img_array2, show_grid=True)
'''
env = Environment(np_img_array,0,0, 19, 19)
#state = env.env_map.copy()
#print(state)
agent = Agent(env)
state = env.env_map.copy()
print(f"{env.env_map}")
env.env_print()
print(env.where_am_i(state))=
state, reward = env.interaction(env.env_map, "left")
print("\n")
env.env_print()
print(f"reward: {reward}")
print("copy initialize agent")
#agent.state_to_key(state)
action = agent.choose_action(state)
print(f"action: {action}")

for episode in range(10):
    done = False
    #state = env.env_map.copy()
    print(state)
    action_idx = agent.choose_action(state)
    action = agent.actions[action_idx]
    next_state, reward = env.interaction(state.copy(), action)
    done = reward == env.goal_reward
    agent.learn((state, action_idx, reward, next_state, done))
    state = next_state.copy()
'''