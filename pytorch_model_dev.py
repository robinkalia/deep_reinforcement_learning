import torch
from torch import nn 


class MyModel(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int, dropout_rate: float=0.1):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_input_features, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(10, num_output_features),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    num_input_features = 20
    num_output_features = 10
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    my_model = MyModel(num_input_features, num_output_features)
    print("MyModel instance:-\n", my_model)
    my_model = my_model.to(device=device)
    sample_input = torch.rand(size=(1, num_input_features), dtype=torch.float32, device=device)
    sample_output = my_model(sample_input)
    sample_output_sum = sample_output.sum()

    print("Sample Input:-\n", sample_input)
    print("\nSample_Output:-\n", sample_output)
    print("\nSample Output Sum =", sample_output_sum)