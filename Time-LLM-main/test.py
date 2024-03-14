import torch
import torch.nn.functional as F

# 假设我们有一个输入张量
input_tensor = torch.tensor([[1.0, 2.0, -1.0],
                             [0.5, -2.0, 3.0]]).to(torch.bfloat16)

# 创建一个掩码，将第二个样本的第一个位置和第一个样本的第三个位置设为 True，表示这些位置需要被掩盖
mask = torch.tensor([[False, True, False],
                     [True, False, False]])

# 将掩码中需要被掩盖的位置替换为负无穷
masked_input = input_tensor.masked_fill(mask, float('-inf')).to(torch.bfloat16)

# 应用 Softmax 函数
softmax_output = F.softmax(masked_input, dim=-1).to(torch.bfloat16)

print("Softmax 输出：", softmax_output)

