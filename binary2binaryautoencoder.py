import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Binary2BinaryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_function=nn.LeakyReLU()):
        super(Binary2BinaryEncoder, self).__init__()
        layers = [nn.Linear(input_dim, hidden_layers[0]), activation_function]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(activation_function)
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        layers.append(nn.Sigmoid())  # Use sigmoid to keep outputs between 0 and 1
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Binary2BinaryDecoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_function=nn.LeakyReLU()):
        super(Binary2BinaryDecoder, self).__init__()
        layers = [nn.Linear(input_dim, hidden_layers[0]), activation_function]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(activation_function)
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Binary2BinaryAutoencoder(nn.Module):
    def __init__(self, input_dim, encoded_dim, encoder_hidden_layers, decoder_hidden_layers, activation_function=nn.LeakyReLU()):
        super(Binary2BinaryAutoencoder, self).__init__()
        self.encoder = Binary2BinaryEncoder(input_dim, encoder_hidden_layers, encoded_dim, activation_function)
        self.decoder = Binary2BinaryDecoder(encoded_dim, decoder_hidden_layers, input_dim, activation_function)

    def forward(self, x, num_fixed=3):
        encoded = self.encoder(x)
        # Use straight-through estimator for training with binary outputs
        encoded_binary = (encoded > 0.5).float().detach() + encoded - encoded.detach()

        # Replace the last 'num_fixed' bits of the encoded output with 0.5
        fixed_values = torch.full((encoded_binary.shape[0], num_fixed), 0.5, device=encoded_binary.device)
        encoded_binary[:, -num_fixed:] = fixed_values.detach()  # Detach to prevent gradients for the fixed part

        decoded = self.decoder(encoded_binary)
        return encoded, decoded


class RewardPredictor(torch.nn.Module):
    def __init__(self, n_actions, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_actions = n_actions

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2 * n_latent_dims + n_actions, 1)])
        else:
            self.layers.extend(
                [torch.nn.Linear(2 * n_latent_dims + n_actions, n_units_per_layer),
                 torch.nn.LeakyReLU(inplace=True), ])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.LeakyReLU(inplace=True), ] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, 1)])

        self.reward_predictor = torch.nn.Sequential(*self.layers)

    def forward(self, z0, z1, a):
        a_logits = F.one_hot(a, num_classes=self.n_actions).float()
        context = torch.cat((z0, z1, a_logits), -1)
        reward = self.reward_predictor(context).squeeze(-1)
        return reward


# Example setup
input_dim = 10
encoded_dim = 4
encoder_hidden_layers = [20, 15]
decoder_hidden_layers = [15, 20]

# Instantiate the autoencoder
autoencoder = Binary2BinaryAutoencoder(input_dim, encoded_dim, encoder_hidden_layers, decoder_hidden_layers)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Example input
x = torch.randn(1, input_dim)  # Generate a random input vector

# Training process
for epoch in range(100):  # Assume training for 100 epochs
    optimizer.zero_grad()
    encoded, decoded = autoencoder(x, num_fixed=2)  # Let's say we fix the last 2 bits
    loss = criterion(decoded, x)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Check the outputs after training
encoded, decoded = autoencoder(x, num_fixed=2)
print("Original:", x)
print("Encoded:", encoded)
print("Decoded:", decoded)
