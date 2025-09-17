import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_features, time_length, num_targets, conv_1_out=64, conv_2_out=128, conv_1_kernel_size=3, \
        conv_2_kernel_size=3, pooling='max', pool_1_kernel_size=2, pool_2_kernel_size=2, hidden_layer_size=256, \
        dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=conv_1_out, kernel_size=conv_1_kernel_size, padding=conv_1_kernel_size//2) #padding so that we dont reduce time dimension (sequence length) at conv layers

        self.bn1 = nn.BatchNorm1d(conv_1_out)

        self.relu = nn.ReLU()

        if pooling == 'max':
            self.pool_1 = nn.MaxPool1d(kernel_size=pool_1_kernel_size)
        elif pooling == 'avg':
            self.pool_1 = nn.AvgPool1d(kernel_size=pool_1_kernel_size)
        else:
            raise ValueError(f'Invalid pooling type: {pooling}')

        self.conv2 = nn.Conv1d(in_channels=conv_1_out, out_channels=conv_2_out, kernel_size=conv_2_kernel_size, padding=conv_2_kernel_size//2)

        self.bn2 = nn.BatchNorm1d(conv_2_out)

        if pooling == 'max':
            self.pool_2 = nn.MaxPool1d(kernel_size=pool_2_kernel_size)
        elif pooling == 'avg':
            self.pool_2 = nn.AvgPool1d(kernel_size=pool_2_kernel_size)
        else:
            raise ValueError(f'Invalid pooling type: {pooling}')

        self.flatten = nn.Flatten()


        with torch.no_grad(): # get input size to fc1 after all layers
            #prints to help visualize how layers change input size
            dummy = torch.zeros(1, num_features, time_length)
            print(f'Input size: batch_size: {dummy.shape[0]}, channels: {dummy.shape[1]}, sequence_length: {dummy.shape[2]}')
            dummy_out = self.conv1(dummy)
            print(f'Conv1 weight shape: {self.conv1.weight.shape}')
            print(f'After conv1: batch_size: {dummy_out.shape[0]}, channels: {dummy_out.shape[1]}, sequence_length: {dummy_out.shape[2]}')
            dummy_out = self.bn1(dummy_out)
            print(f'After bn1: batch_size: {dummy_out.shape[0]}, channels: {dummy_out.shape[1]}, sequence_length: {dummy_out.shape[2]}')
            dummy_out = self.relu(dummy_out)
            print(f'After relu: batch_size: {dummy_out.shape[0]}, channels: {dummy_out.shape[1]}, sequence_length: {dummy_out.shape[2]}')
            dummy_out = self.pool_1(dummy_out)
            print(f'After pool1: batch_size: {dummy_out.shape[0]}, channels: {dummy_out.shape[1]}, sequence_length: {dummy_out.shape[2]}')

            dummy_out = self.conv2(dummy_out)
            print(f'Conv2 weight shape: {self.conv2.weight.shape}')
            print(f'After conv2: batch_size: {dummy_out.shape[0]}, channels: {dummy_out.shape[1]}, sequence_length: {dummy_out.shape[2]}')
            dummy_out = self.bn2(dummy_out)
            print(f'After bn2: batch_size: {dummy_out.shape[0]}, channels: {dummy_out.shape[1]}, sequence_length: {dummy_out.shape[2]}')
            dummy_out = self.relu(dummy_out) # Add this line
            print(f'After relu2: batch_size: {dummy_out.shape[0]}, channels: {dummy_out.shape[1]}, sequence_length: {dummy_out.shape[2]}')
            dummy_out = self.pool_2(dummy_out)
            print(f'After pool2: batch_size: {dummy_out.shape[0]}, channels: {dummy_out.shape[1]}, sequence_length: {dummy_out.shape[2]}')
            dummy_out = self.flatten(dummy_out)
            print(f'After flatten: batch_size: {dummy_out.shape[0]}, features: {dummy_out.shape[1]}')

            flatten_size = dummy_out.view(1, -1).size(1)

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=hidden_layer_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=num_targets) #multivariate


    def forward(self, x):
        # Debug input
        #print(f"Input stats - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")
        
        x = x.to(self.conv1.bias.dtype)  # needs to be same dtype as bias tensor
        #print(f"After dtype conversion - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")

        x = x.permute(0, 2, 1)  # (batch_size, in_channels, sequence_length)
        #print(f"After permute - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")

        x = self.conv1(x)
        #print(f"After conv1 - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")
        
        x = self.bn1(x)
        #print(f"After bn1 - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")
        
        x = self.relu(x)
        x = self.pool_1(x)
        #print(f"After pool1 - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")
        
        x = self.conv2(x)
        #print(f"After conv2 - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")
        
        x = self.bn2(x)
        #print(f"After bn2 - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")
        
        x = self.relu(x)
        x = self.pool_2(x)
        #print(f"After pool2 - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")
        
        x = self.flatten(x)
        x = self.fc1(x)
        #print(f"After fc1 - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")
        
        x = self.dropout(x)
        x = self.fc2(x)
        #print(f"Final output - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, isnan: {torch.isnan(x).any()}")

        return x