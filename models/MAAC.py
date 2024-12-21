import torch.nn as nn
import torch


class MAAC(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        """ 
        Sử dụng 3 tầng convolutional:
            Lớp 1: 32 kênh (filters).
            Lớp 2: 64 kênh.
            Lớp 3: 128 kênh.
        ReLU: Hàm kích hoạt phi tuyến giúp mạng học các biểu diễn phức tạp hơn.
        Padding=1: Giữ nguyên kích thước không gian sau mỗi tầng tích chập.
        """
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Tính toán kích thước flatten đầu ra từ CNN
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        
        # Actor network for selecting actions
        # Actor network: Dự đoán hành động dựa trên đặc trưng đã trích xuất từ trạng thái.
        """ 
        Mục tiêu: Sinh các xác suất hoặc hành động từ đặc trưng đầu ra CNN.
            nn.Linear(flatten_dim, 256): Nhận đầu vào từ CNN và giảm xuống không gian 256 chiều.
            action_shape: Đầu ra số lượng hành động (một giá trị duy nhất hoặc xác suất cho từng hành động).
        """
        self.actor = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_shape), 
        )
        
        # Critic network for evaluating state-action pairs
        # Critic network: Đánh giá giá trị trạng thái-hành động (Q-value) để tối ưu chính sách.
        """ 
        Mục tiêu: Tính toán giá trị Q cho cặp (trạng thái, hành động).
            Nhận đầu vào là đặc trưng của trạng thái (CNN output) cộng với one-hot encoding của hành động.
            Đầu ra là giá trị Q duy nhất.
        """
        self.critic = nn.Sequential(
            nn.Linear(flatten_dim + action_shape, 256),  
            nn.ReLU(),
            nn.Linear(256, 1), 
        )
    # thực hiện dự đoán hành động từ quan sát.
    """ 
    Hàm forward thực hiện các bước sau:

        Kiểm tra đầu vào để đảm bảo đúng định dạng.
        Trích xuất đặc trưng từ observation bằng mạng CNN.
        Xác định kích thước batch và chuyển đổi đặc trưng thành vector phẳng.
        Tính toán xác suất hoặc giá trị hành động qua mạng actor.
        Trả về kết quả hành động.
    """
    def forward(self, observation):
        assert len(observation.shape) >= 3, "Only support Magent input observation"
        """ 
        Xử lý quan sát bằng CNN
            Quan sát được đưa qua mạng CNN để trích xuất đặc trưng.
            Nếu quan sát không theo dạng batch (kích thước 3D), giả sử kích thước batch là 1
            Đầu ra từ actor là vector các xác suất hành động hoặc các giá trị hành động.
        """
        x = self.cnn(observation)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)

        action_probs = self.actor(x)

        return action_probs
    
    # Hàm này đánh giá giá trị Q cho trạng thái và hành động đầu vào.
    """ 
    Hàm evaluate thực hiện các bước sau:

        Trích xuất đặc trưng từ trạng thái đầu vào (observation) bằng mạng CNN.
        Tính xác suất hành động từ mạng actor.
        Chuyển đổi hành động thành one-hot encoding.
        Ghép nối đặc trưng và one-hot encoding, rồi đưa qua mạng critic để tính Q-value.
        Trả về Q-value và xác suất hành động.
    """
    def evaluate(self, observation, action):
        """ Evaluate Q-value for given state-action pair """
        assert len(observation.shape) >= 3, "Only support Magent input observation"
        
        # Trích xuất đặc trưng từ observation
        x = self.cnn(observation)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)

        action_probs = self.actor(x)
        
        action_one_hot = torch.zeros(batchsize, action_probs.shape[1], device=observation.device)
        action_one_hot.scatter_(1, action.unsqueeze(1), 1)  
        
        state_action_value = self.critic(torch.cat([x, action_one_hot], dim=1))
        
        return state_action_value, action_probs