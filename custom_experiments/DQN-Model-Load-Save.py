import os
import sys
from datetime import datetime

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from sumo_rl import SumoEnvironment


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


no_of_episodes = 5
# generate number of routes, then iterate over them while training


if __name__ == "__main__":

    now = datetime.now()
    local_time = now.strftime("%d-%m-%Y_%H-%M-%S")

    for e in range(no_of_episodes) :

        env = SumoEnvironment(
            net_file="../custom_nets/2WSI/single-intersection.net.xml",
            route_file=f"../custom_nets/2WSI/uniform_{e + 1}.rou.xml",
            out_csv_name=f"../outputs/2WSI/DQN_{local_time}_Episode_{e + 1}",
            single_agent=True,
            use_gui=True,
            num_seconds=5400,
        )

        if e > 0 :
            model = DQN.load(f"../models/DQN_{local_time}_Episode_{e}",env = env)

        else :
            model = DQN(
                env=env,
                policy="MlpPolicy",
                learning_rate=0.001,
                learning_starts=0,
                train_freq=1,
                target_update_interval=200,
                exploration_initial_eps=0.05,
                exploration_final_eps=0.01,
                verbose=1,
            )
        
        model.learn(total_timesteps=(5400)/5)
        model.save(f"../models/DQN_{local_time}_Episode_{e + 1}")
        env.close()


    # env = SumoEnvironment(
    #         net_file="nets/2way-single-intersection/single-intersection.net.xml",
    #         route_file=f"nets/2way-single-intersection/train/uniform_1.rou.xml",
    #         out_csv_name=f"outputs/2way-single-intersection/dqn_episode_test_2",
    #         single_agent=True,
    #         use_gui=True,
    #         num_seconds=5400,
    #     )    

    # model = DQN.load(f"modified_dqn 5",env = env)
    # model.learn(total_timesteps=(5400)/5)
    # model.save(f"modified_dqn_test 2")
    # env.close()