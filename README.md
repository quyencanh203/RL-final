# Dự Án Cuối Khóa MAgent2 RL
## Tổng Quan
Trong dự án cuối khóa này, bạn sẽ phát triển và huấn luyện một tác nhân học tăng cường (RL) sử dụng nền tảng MAgent2. Nhiệm vụ là giải quyết môi trường MAgent2 được chỉ định `battle`, và tác nhân đã huấn luyện của bạn sẽ được đánh giá trên ba loại đối thủ sau:

1. Tác nhân Ngẫu Nhiên: Tác nhân thực hiện các hành động ngẫu nhiên trong môi trường.
2. Tác nhân Được Huấn Luyện Trước: Một tác nhân đã được huấn luyện có sẵn trong kho lưu trữ.

Tác nhân xanh cạnh tranh với các tác nhân đỏ ngẫu nhiên, trong khi bên phải hiển thị một cuộc chiến giữa hai tác nhân tự chơi. Các tác nhân xanh có khả năng đánh bại tác nhân ngẫu nhiên, cho thấy khả năng của chúng đối với các tác nhân chưa được huấn luyện, nhưng gặp khó khăn với các tác nhân đỏ, được huấn luyện nhiều hơn để vượt trội hơn các tác nhân xanh. Như trước đây, bạn nên đánh giá tác nhân của mình trước các tác nhân đỏ.

## Cài Đặt
- Clone dự án với [link github](https://github.com/quyencanh203/RL-final)
- pip install -r requirements.txt

## training với colab
[Link colab](https://colab.research.google.com/drive/1b3HEHgyadmkQmEBO6m4j7ev7uGYDIBcT?usp=sharing)

## Demo
Xem `demo.py` để chạy demo.

## Đánh Giá
Sử dụng `eval.py` để đánh giá model pretrained.
{'winrate_red', 'winrate_blue', 'average_rewards_red', 'average_rewards_blue'}

## Thành viên
- Phạm Anh Quân
- Hà Kim Dương 
- Hồ Cảnh Quyền 
## Tài Liệu Tham Khảo

1. [Kho Lưu Trữ GitHub MAgent2](https://github.com/Farama-Foundation/MAgent2)
2. [Tài Liệu API MAgent2](https://magent2.farama.org/introduction/basic_usage/)
3. [github anh giang](https://github.com/giangbang/RL-final-project-AIT-3007)
