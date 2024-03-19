from utils.losses import smape_loss, mask_Metrics, mape_loss
import torch




a = torch.tensor([1]).repeat(10).reshape(2, 5, 1)
b = torch.tensor([2]).repeat(10).reshape(2, 5, 1)
m = torch.tensor([1,1,1,1,0,1,1,1,0,0]).reshape(2,5,1)


l1 = smape_loss()(0,0,a,b,m)/200
l2 = mask_Metrics()(a,b,m)

print(l1)
print(l2)

