import torch

x = torch.tensor([2.0])
y_true = torch.tensor([5.0])

w = torch.tensor([0.0], requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

prediction = w * x
loss = (prediction - y_true).pow(2).mean()

loss.backward()       # หา gradient ของ loss ต่อ w
optimizer.step()      # ปรับ w
optimizer.zero_grad() # ล้าง gradient เก่าสำหรับรอบถัดไป

print(w)