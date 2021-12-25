import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import tqdm

from LSTM import LSTM
from dataset import FaceDataset

epoches = 200
batch_size = 8
time_step = 28
input_size = 28
learning_rate = 0.05
hidden_size = 64
num_layers = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


f = open("final_data_list.txt", "r")
lines = f.readlines()
f.close()
lines = [line.strip().replace("\\", "/") for line in lines]
train_X, test_X, _, _ = train_test_split(lines, range(len(lines)), test_size=0.3, random_state=17)
train_set = FaceDataset(files_path=train_X)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = FaceDataset(files_path=test_X)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


def main():
    lstm = LSTM().to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    lstm_loss = nn.CrossEntropyLoss().to(device)

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
                total_acc += (predicted == label).sum()
                loss.backward()
                optimizer.step()
                total_iter = total_iter + batch_size
                t.set_postfix(
                    loss=total_loss / total_iter,
                    acc=total_acc.cpu().numpy() / total_iter,
                )
                t.update(1)
        with torch.no_grad():
            y_pred = None
            y_true = None
            acc = 0
            total = 0
            for (feature_test, label_test) in test_loader:
                feature_test, label_test = feature_test.to(device), label_test.to(device)
                outputs = lstm(feature_test)
                _, predicted = torch.max(outputs.data, 1)
                total += label_test.size(0)
                y_pred = (
                    predicted.data
                    if y_pred == None
                    else torch.cat((y_pred, predicted.data))
                )
                y_true = (
                    label_test
                    if y_true == None
                    else torch.cat((y_true, label_test))
                )
                acc += (predicted == label_test).sum()
            print(
                confusion_matrix(
                    y_true=y_true.cpu().detach().numpy(),
                    y_pred=y_pred.cpu().detach().numpy(),
                )
            )
            acc = 100 * acc / total
            print(acc.item(), "%")


if __name__ == "__main__":
    main()
