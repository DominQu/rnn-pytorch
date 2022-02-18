import torch
from torch import nn
from torch.optim import Adam
from torchinfo import summary
from pathlib import Path
import matplotlib.pyplot as plt
import time


class DataLoader:
    def __init__(self, path, trainprc):
        self.path = Path(path)
        self.trainprc = trainprc
        with self.path.open() as file:
            self.data = file.read()
            self.data = self.data.lower()
            self.chars = list(set(self.data))
            self.chars.sort()
            self.charnum = len(self.chars)
        self.dataptr = 0

    def showchars(self):
        print(self.chars)

    def encode(self):
        self.dataset = torch.zeros((len(self.data), self.charnum))
        for charidx, char in enumerate(self.data):
            ind = self.chars.index(char)
            onehot = torch.zeros((1, self.charnum))
            onehot[0, ind] = 1
            self.dataset[charidx] = onehot

    def decode(self, onehot):
        arg = torch.argmax(onehot)
        return self.chars[arg]

    def getchar(self, index):
        return self.chars[index]

    def randomsample(self):
        ran = torch.randint(0, len(self.data), (1,)).item()
        print(self.dataset[ran])
        print(self.decode(self.dataset[ran]))

    def todevice(self, device):
        self.dataset.to(device)

    def getBatch(self, batchsize, timesteps):
        randindices = torch.randint(
            0, int(self.trainprc * len(self.data)) - timesteps - 1, (batchsize,)
        )
        batchelems = tuple(
            self.dataset[randindices[i] : randindices[i] + timesteps, :]
            for i in range(batchsize)
        )
        targets = tuple(
            self.dataset[randindices[i] + 1 : randindices[i] + timesteps + 1, :]
            for i in range(batchsize)
        )
        return torch.stack(batchelems, dim=1), torch.stack(targets, dim=1)

    def getTestBatch(self, batchsize, timesteps):
        randindices = torch.randint(
            int(self.trainprc * len(self.data)),
            len(self.data) - timesteps - 1,
            (batchsize,),
        )
        batchelems = tuple(
            self.dataset[randindices[i] : randindices[i] + timesteps, :]
            for i in range(batchsize)
        )
        targets = tuple(
            self.dataset[randindices[i] + 1 : randindices[i] + timesteps + 1, :]
            for i in range(batchsize)
        )
        return torch.stack(batchelems, dim=1), torch.stack(targets, dim=1)

    def getIndices(self, onehots):
        indices = []
        for batch in onehots:
            batchind = []
            for encoding in batch:
                arg = torch.argmax(encoding)
                batchind.append(arg)
            indices.append(batchind)
        return torch.tensor(indices).permute((1, 0))


class RNN(nn.Module):
    def __init__(self, input_size, state_size, layers=1, dropout=0):
        super(RNN, self).__init__()
        self.stat_size = state_size
        self.layers = layers

        self.LSTM = nn.LSTM(input_size, state_size, num_layers=layers, dropout=dropout)
        self.outlinear = nn.Linear(state_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.loss = nn.NLLLoss()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, state):
        lstmout, (h, c) = self.LSTM(input, state)
        linearout = self.outlinear(lstmout)
        dropout = self.dropout(linearout)
        # probs = self.softmax(dropout)
        return dropout


def train(state_size, batchsize, timesteps, learning_rate):
    hidden = torch.zeros((rnn.layers, batchsize, state_size))
    carry = torch.zeros((rnn.layers, batchsize, state_size))
    optimizer.zero_grad()
    input, targets = dataloader.getBatch(batchsize, timesteps)
    targets = dataloader.getIndices(targets)
    outputs = rnn(input, (hidden, carry))

    loss = rnn.loss(torch.permute(outputs, (1, 2, 0)), targets)

    loss.backward()
    optimizer.step()

    return outputs, loss.item()


def test(state_size, batchsize, timesteps):
    hidden = torch.zeros((rnn.layers, batchsize, state_size))
    carry = torch.zeros((rnn.layers, batchsize, state_size))
    input, targets = dataloader.getTestBatch(batchsize, timesteps)
    targets = dataloader.getIndices(targets)
    outputs = rnn(input, (hidden, carry))
    loss = rnn.loss(torch.permute(outputs, (1, 2, 0)), targets)

    return outputs, loss.item()


def topk(distribution, k):
    result = torch.zeros_like(distribution)
    indices = torch.topk(torch.exp(distribution), k)[1]
    result[0, indices] = torch.exp(distribution)[0, indices]
    return result


def generatetext(text_lenght, timesteps):
    hidden = torch.zeros((rnn.layers, 1, state_size))
    carry = torch.zeros((rnn.layers, 1, state_size))
    input, _ = dataloader.getBatch(1, timesteps)
    for t in range(timesteps):
        print(dataloader.decode(input[t]), end="")
    print("\nGenerated text: ")
    for t in range(text_lenght):
        outputs = rnn(input, (hidden, carry))
        inputcopy = input.clone()
        input[0:-1, :] = inputcopy[1:, :]
        input[-1, :] = rnn.softmax(outputs[-1])
        newelem = torch.zeros_like(input[-1])

        prob = topk(input[-1], 5)
        ind = torch.multinomial(prob, 1).item()
        letter = dataloader.getchar(ind)
        newelem[0, ind] = 1.0
        input[-1, :] = newelem
        print(letter, end="")


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # path = "/home/dominik/cudadir/rnn-cuda/data/dziady-ascii.txt"
    # path = "data/shakespeare.txt"
    path = "data/anna.txt"

    trainprc = 0.9
    dataloader = DataLoader(path, trainprc)
    dataloader.encode()
    # dataloader.showchars()
    dataloader.todevice(device)

    # Network parameters
    input_size = dataloader.charnum
    state_size = 512
    layers = 1
    dropout = 0.5
    learning_rate = 0.1
    batchsize = 64
    timesteps = 64
    epochs = 5000
    data = "dz"
    model = f"models/{data}_layers_{layers}_batch_{batchsize}_dropout_{dropout}_state_{state_size}_timesteps_{timesteps}"
    log = 10
    load = False
    loadedloss = 2.474

    rnn = RNN(input_size, state_size, layers, dropout)
    summary(rnn)
    optimizer = Adam(rnn.parameters())

    if load:
        rnn.load_state_dict(torch.load(model + f"_loss_{loadedloss}.pth"))

    losshistory = []
    testlosshistory = []

    start = time.perf_counter()
    # training loop
    for i in range(epochs):
        epochstart = time.perf_counter()
        outputs, loss = train(state_size, batchsize, timesteps, learning_rate)
        if i % log == 0:
            _, testloss = test(state_size, batchsize, timesteps)
            testlosshistory.append(testloss)
            losshistory.append(loss)
            minutes = (time.perf_counter() - start) // 60
            print(
                f"Epoch: {i}, Current loss: {loss:0.4f}, Test loss: {testloss:0.4f}, training duration: {minutes:0.0f} min \
{(time.perf_counter() - start) - 60*minutes:0.2f} sec"
            )

    stop = time.perf_counter()
    minutes = (stop - start) // 60
    seconds = (stop - start) - 60 * minutes
    print(f"Training time: {minutes:0.0f} min {seconds:0.4f} sec")
    generatetext(300, timesteps)
    
    plt.plot(losshistory)
    plt.plot(testlosshistory)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(("Train loss", "Test loss"))
    plt.show()
    model += f"_loss_{round(losshistory[-1], 3)}.pth"
    torch.save(rnn.state_dict(), model)
    print(f"\nSaved PyTorch Model State to {model}")
