from magent2.environments import battle_v4
import torch
import os
import cv2
from models.torch_model import QNetwork
from models.final_torch_model import FinalQNetwork
from models.mymodel import ActorCriticModel
import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 35
    frames = []
    
    
    blue_q_network0 = ActorCriticModel(
    env.observation_space("blue_0").shape, env.action_space("blue_0").n
    )
    blue_q_network0.load_state_dict(
        torch.load("pretrains/bluemaac.pt", weights_only=True, map_location=device)
    )
    # random policies
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "blue":
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = blue_q_network0(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                action = env.action_space(agent).sample()

        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"random.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording random agents")

    # pretrain_1
    frames = []
    env.reset()
    blue_q_network = ActorCriticModel(
    env.observation_space("blue_0").shape, env.action_space("blue_0").n
    )
    blue_q_network.load_state_dict(
        torch.load("pretrains/bluemaac.pt", weights_only=True, map_location=device)
    )

    red_q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    red_q_network.load_state_dict(
        torch.load("pretrains/red.pt", weights_only=True, map_location=device)
    )
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "red":
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = red_q_network(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                # action = env.action_space(agent).sample()
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = blue_q_network(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
                
        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"pretrained.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording pretrained agents")

    env.close()

    # pretrained policies (pretrain_final)
    frames = []
    env.reset()
    blue_q_network = ActorCriticModel(
    env.observation_space("blue_0").shape, env.action_space("blue_0").n
    )
    blue_q_network.load_state_dict(
        torch.load("pretrains/bluemaac.pt", weights_only=True, map_location=device)
    )

    red_q_network = FinalQNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    red_q_network.load_state_dict(
        torch.load("pretrains/red_final.pt", weights_only=True, map_location=device)
    )
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "red":
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = red_q_network(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                # action = env.action_space(agent).sample()
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = blue_q_network(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
                
        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"pretrained_final.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording pretrained_final agents")

    env.close()