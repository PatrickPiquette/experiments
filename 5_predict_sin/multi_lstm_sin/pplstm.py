import torch
import torch.nn as nn

from hidden import Hidden


class PPLSTM(nn.Module):

    def __init__(self, hidden_size, dropout=0.05, num_layers=2, mode="LSTM", device="cpu"):
        super(PPLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_size = 1
        self.lstm = nn.ModuleList()
        self.hiddens = []
        self.mode = mode
        self.training = False
        self.device = device

        assert num_layers >= 1

        if self.mode is "LSTM":
            self.lstm.append(nn.LSTM(input_size=1,
                                     hidden_size=hidden_size,
                                     num_layers=self.num_layers,
                                     dropout=self.dropout))
            self.hiddens.append(Hidden(self.num_layers, self.batch_size, self.hidden_size, device=device))

        elif self.mode is "LSTMCell":
            self.lstm.append(nn.LSTMCell(input_size=1, hidden_size=hidden_size))

            for layer in range(self.num_layers - 1):
                self.lstm.append(nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size))

            for layer in range(self.num_layers):
                self.hiddens.append(Hidden(self.batch_size, self.hidden_size, device=device))

        else:
            raise ValueError(f"Unsuported mode {mode}")

        self.out = nn.Linear(hidden_size, 1)

    def reset_state(self):
        for hidden in self.hiddens:
            hidden.reset_state()

    def printParams(self, index=None):
        if index is None:
            print("\nPrinting parameters")
            for parameter in self.parameters():
                print(parameter)
        else:
            for i, value in enumerate(self.parameters()):
                if i == index:
                    print(f"\nPrinting parameters at index {i}")
                    print(value)
                    break

    def evaluate(self, inputs):
        nbr_inputs = inputs.size()[0]
        if self.mode is "LSTM":
            o = torch.zeros(nbr_inputs, 1, 1).to(self.device)
            o, hidden = self.lstm[0](inputs, self.hiddens[0].get_h_c())
            self.hiddens[0].set_h_c(hidden)
            o = self.out(o)
        else:
            o = torch.zeros(nbr_inputs, 1).to(self.device)
            for i, input in enumerate(inputs):
                # lstm layer 0
                hidden = self.lstm[0](input.unsqueeze(1), self.hiddens[0].get_h_c())
                self.hiddens[0].set_h_c(hidden)

                # all subsequent lstm layers
                for layer in range(1, self.num_layers):
                    hidden = self.lstm[layer](self.hiddens[layer - 1].h_0, self.hiddens[layer].get_h_c())
                    self.hiddens[layer].set_h_c(hidden)

                # linear taking the last layer h_0 as input
                o[i] = self.out(self.hiddens[self.num_layers - 1].h_0)

            if self.training == True:
                self.reset_state()
        return (o)

    def forward(self, inputs, training=False):
        self.training = training

        # Rearrange the inputs to the correct size based on the selected mode
        if len(inputs.size()) == 1:
            inputs.unsqueeze_(1)
            if self.mode == "LSTM":
                inputs.unsqueeze_(2)

        # Define local variables based on the selected mode
        if self.mode == "LSTM":
            input = torch.zeros(1, 1, 1).to(self.device)
            input[0, 0, 0] = inputs[0, 0, 0]
            outputs = torch.zeros(inputs.size()[0], 1, 1).to(self.device)
        else:
            input = torch.zeros(1, 1).to(self.device)
            input[0, 0] = inputs[0, 0]
            outputs = torch.zeros(inputs.size()[0], 1).to(self.device)

        if self.training is True:
            self.reset_state()
            outputs = self.evaluate(inputs)
        else:
            # In predicting mode
            # We feed the first elements of inputs for the first loop and then feed the model output into the input.
            for i in range(inputs.size()[0]):
                output = self.evaluate(input)
                outputs[i] = output
                input = output
            self.reset_state()
        return outputs
