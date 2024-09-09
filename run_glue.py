from sub_model import Sub_model
from dataset import MyDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


do_train = True
num_epochs = 10
data_size = 1024
test_size = 128

#make dummy data
data = torch.randn(data_size, 2)
tag = []
for input in data:
    i,j = input
    if i > j:
        tag.append(0)
    else:
        tag.append(1)
labels = torch.tensor(tag)
dataset = MyDataset(data, labels)
train_loader = DataLoader(dataset, batch_size = 32, shuffle = True)

data = torch.randn(test_size, 2)
tag = []
for input in data:
    i,j = input
    if i > j:
        tag.append(0)
    else:
        tag.append(1)
labels = torch.tensor(tag)
dataset = MyDataset(data, labels)
test_loader = DataLoader(dataset, batch_size = 32, shuffle = True)



model = Sub_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#training

if not do_train:
    print('no training on this round')
else:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}')
    print('Training finished')

print('testing')

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        input, labels = data
        outputs = model(input)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total}')
        

torch.save(model.state_dict(), 'model.pt')

