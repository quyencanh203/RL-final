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
from models.MAAC import MAAC

def train_model(env, blue_q_network, red_q_network, optimizer, replay_buffer, device, args):
    epsilon = args.epsilon_start
    steps_done = 0
    losses = []

    for episode in range(args.max_episodes):
        state = env.reset()
        episode_reward = 0  # Tổng phần thưởng của agent xanh trong mỗi episode.

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
                next_observation, reward, termination, truncation, _ = env.last()  # Lấy trạng thái và phần thưởng mới
                next_observation = torch.tensor(next_observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                replay_buffer.push(
                    observation.squeeze(0).cpu().numpy(),
                    action,
                    reward,
                    next_observation.squeeze(0).cpu().numpy(),
                    termination or truncation,
                )

                episode_reward += reward

                # Huấn luyện mạng với batch từ Replay Buffer
                if len(replay_buffer) >= args.batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(args.batch_size)
                    states, actions, rewards, next_states, dones = (
                        torch.tensor(states).to(device),
                        torch.tensor(actions).to(device),
                        torch.tensor(rewards).to(device),
                        torch.tensor(next_states).to(device),
                        torch.tensor(dones).to(device),
                    )
                    q_values = blue_q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_q_values = blue_q_network(next_states).max(1)[0]
                    target_q_values = rewards + args.gamma * next_q_values * (1 - dones)
                    loss = nn.MSELoss()(q_values, target_q_values)

                    losses.append(loss.item())
                    print(f"Episode {episode + 1}, Step {steps_done}, Loss: {loss.item()}")

                    # Cập nhật các tham số của mạng Q
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Agent đỏ sử dụng mạng red_q_network đã được huấn luyện trước đó
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

        print(f"Episode {episode + 1}/{args.max_episodes}, Reward: {episode_reward}")

        # Lưu mạng và in kết quả
        if (episode + 1) % args.target_update_freq == 0:
            torch.save(blue_q_network.state_dict(), f"blue_agent_episode_{episode + 1}.pt")

    return losses


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

    red_q_network = FinalQNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
    state_dict = torch.load("pretrains/red_final.pt", map_location=device, weights_only=True)
    red_q_network.load_state_dict(state_dict)
    red_q_network.eval()

    blue_q_network = MAAC(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    optimizer = optim.Adam(blue_q_network.parameters(), lr=args.learning_rate)

    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    losses = train_model(env, blue_q_network, red_q_network, optimizer, replay_buffer, device, args)
    torch.save(blue_q_network.state_dict(), "pretrains/bluemaac.pt")
    print("Saved model!")
    env.close()

