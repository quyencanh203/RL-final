import torch
import torch.nn as nn

class my_QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape, padding=1):
        super().__init__()

        input_channels = observation_shape[-1]

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=padding),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=padding),
            nn.ReLU(),
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=padding),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=padding),
            nn.ReLU(),
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, padding=0),
            nn.ReLU()
        )

        # Định nghĩa các lớp fully connected
        self.fc = nn.Sequential(
            nn.Linear(self._get_flatten_dim(observation_shape), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_shape)
        )

    def _get_flatten_dim(self, observation_shape):
        """Giúp tính toán số lượng tham số sau khi flatten (sau các lớp Conv2d)."""
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)  # Chuyển đổi thành [C, H, W]
        dummy_output = self.cnn(dummy_input)
        return dummy_output.numel()  # Trả về số lượng phần tử trong tensor sau khi qua CNN

    def forward(self, x):
        assert len(x.shape) >= 3, "Only support input with at least 3 dimensions"

        # Tiến hành chập qua CNN
        x = self.cnn(x)

        # Flatten output từ CNN
        batch_size = x.shape[0] if len(x.shape) > 2 else 1
        x = x.view(batch_size, -1)

        # Đưa output vào các lớp fully connected
        return self.fc(x)

