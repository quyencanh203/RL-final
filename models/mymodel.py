import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = self.global_pool(x)
        return x.flatten(start_dim=1)


class ActorCriticModel(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(observation_shape[-1])

        flatten_dim = 128  # Sau khi global pool
        self.actor = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, action_shape),
        )

        self.critic = nn.Sequential(
            nn.Linear(flatten_dim + action_shape, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )

    def forward(self, observation, return_logits=False):
        assert len(observation.shape) == 4, "Input must have shape (batch_size, channels, height, width)"
        x = self.feature_extractor(observation)
        action_probs = self.actor(x)
        if return_logits:
            return action_probs
        return torch.softmax(action_probs, dim=-1)

    def evaluate(self, observation, action):
        x = self.feature_extractor(observation)
        action_probs = self.actor(x)

        action_one_hot = torch.nn.functional.one_hot(action, num_classes=action_probs.shape[1]).float()
        action_one_hot = action_one_hot.to(observation.device)

        state_action_value = self.critic(torch.cat([x, action_one_hot], dim=1))
        return state_action_value, action_probs
