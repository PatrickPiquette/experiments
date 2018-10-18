import torch


class Hidden():
    def __init__(self, *args, device):
        self.args = args
        self.device = device
        self.h_0 = self.__default_state()
        self.c_0 = self.__default_state()

    def __default_state(self):
        return torch.zeros(self.args, device=self.device, requires_grad=False)

    def reset_state(self):
        self.h_0 = self.__default_state()
        self.c_0 = self.__default_state()

    def get_h_c(self):
        return self.h_0, self.c_0

    def set_h_c(self, tup):
        self.h_0 = tup[0]
        self.c_0 = tup[1]

