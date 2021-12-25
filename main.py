import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
import tqdm

from LSTM import LSTM
from dataset import FaceDataset

epoches = 2
batch_size = 32
time_step = 28
input_size = 28
learning_rate = 0.01
hidden_size = 64
num_layers = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

train_set = FaceDataset(files_path=xxx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = FaceDataset(files_path=xxx)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)



def main():
    lstm = LSTM().to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    lstm_loss = nn.CrossEntropyLoss()

    for epoch in range(epoches):
        with tqdm.tqdm(total=int(len(train_set) / batch_size), desc="Training") as t:
            total_loss = 0
            total_iter = 0
            total_acc = 0
            for i, (feature, label) in enumerate(train_loader):
                feature, label = feature.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = lstm(feature)
                loss = lstm_loss(outputs, label)
                total_loss += loss.cpu().detach().numpy()
                _, predicted = torch.max(outputs.data, 1)
                total_acc += (predicted == label.argmax(1)).sum()
                loss.backward()
                optimizer.step()
                total_iter = total_iter + batch_size
                t.set_postfix(
                    loss=total_loss / total_iter,
                    acc=total_acc.cpu().numpy() / total_iter,
                )
                t.update(1)
        with torch.no_grad():
            acc = 0
            total = 0
            for (feature_test, label_test) in test_loader:
                feature_test, label_test = feature_test.to(device), label_test.to(device)
                outputs = lstm(feature_test)
                _, predicted = torch.max(outputs.data, 1)
                total += label_test.size(0)
                acc += (predicted == label_test.argmax(1)).sum()
            acc = 100 * acc / total
            print(acc.item(), "%")

if __name__ == "__main__":
    main()

