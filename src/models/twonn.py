import torch



class TwoNN(torch.nn.Module): # McMahan et al., 2016; 199,210 parameters
    def __init__(self, resize, hidden_size, num_classes):
        super(TwoNN, self).__init__()
        self.in_features = resize**2
        self.num_hiddens = hidden_size
        self.num_classes = num_classes
        
        self.features = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=self.in_features, out_features=self.num_hiddens, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=self.num_hiddens, out_features=self.num_hiddens, bias=True),
            torch.nn.ReLU(True)
        )
        self.classifier = torch.nn.Linear(in_features=self.num_hiddens, out_features=self.num_classes, bias=True)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
