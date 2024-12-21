# train_blue
import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from magent2.environments import battle_v4
from Buffers import ReplayBuffer
from models.torch_model import QNetwork
from models.final_torch_model import FinalQNetwork
from models.mymodel import my_QNetwork

def train_model(env, blue_q_network, red_q_network, optimizer, replay_buffer, device, args):
    epsilon = args.epsilon_start
    steps_done = 0

    for episode in range(args.max_episodes):
        state = env.reset()
        episode_reward = 0  # Tổng phần thưởng của agent xanh trong mỗi episode
        episode_losses = []  # Lưu loss của từng step trong 1 episode

        for agent in env.agent_iter():  # Lặp qua từng agent
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            if agent.startswith("blue"):
                observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # ----- epsilon ------
                if np.random.rand() < epsilon:
                    action = env.action_space(agent).sample()
                else:
                    with torch.no_grad():
                        q_values = blue_q_network(observation)
                    action = torch.argmax(q_values, dim=1).item()

                env.step(action)  # Thực hiện hành động
                next_observation, reward, termination, truncation, _ = env.last()
                next_observation = torch.tensor(next_observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                replay_buffer.push(
                    observation.squeeze(0).cpu().numpy(),
                    action,
                    reward,
                    next_observation.squeeze(0).cpu().numpy(),
                    termination or truncation,
                )

                episode_reward += reward
                
                # Huấn luyện mạng Q-learning cho agent "blue"
                if len(replay_buffer) >= args.batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(args.batch_size)
                    states, actions, rewards, next_states, dones = (
                        states.to(device), actions.to(device), rewards.to(device), 
                        next_states.to(device), dones.to(device)
                    )
                    q_values = blue_q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_q_values = blue_q_network(next_states).max(1)[0]
                    target_q_values = rewards + args.gamma * next_q_values * (1 - dones)
                    loss = nn.MSELoss()(q_values, target_q_values)
                    episode_losses.append(loss.item())  # Lưu loss của step hiện tại
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            elif agent.startswith("red"): 
                observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = red_q_network(observation)
                action = torch.argmax(q_values, dim=1).item()
                env.step(action)
            else:
                env.step(env.action_space(agent).sample())
        
        # Giảm giá trị epsilon dần dần
        epsilon = max(args.epsilon_end, args.epsilon_start - (steps_done / args.epsilon_decay))
        steps_done += 1

        # Tính average loss và in ra
        if episode_losses:
            ave_loss = sum(episode_losses) / len(episode_losses)
        else:
            ave_loss = 0.0
        print(f"Episode {episode + 1}/{args.max_episodes}, Reward: {episode_reward}, Average Loss: {ave_loss:.4f}")
        
        # Lưu mạng và in kết quả
        if (episode + 1) % args.target_update_freq == 0:
            torch.save(blue_q_network.state_dict(), f"blue_agent_episode_{episode + 1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    parser.add_argument("--epsilon_decay", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--max_episodes", type=int, default=5)
    parser.add_argument("--replay_buffer_size", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    env.reset()

    red_q_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
    state_dict = torch.load("pretrains/red.pt", map_location=device, weights_only=True)
    red_q_network.load_state_dict(state_dict)
    red_q_network.eval()

    blue_q_network = my_QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    optimizer = optim.Adam(blue_q_network.parameters(), lr=args.learning_rate)

    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    train_model(env, blue_q_network, red_q_network, optimizer, replay_buffer, device, args)
    torch.save(blue_q_network.state_dict(), "pretrains/blue.pt")
    print("Saved model!")
    env.close()

