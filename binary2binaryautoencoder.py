import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Binary2BinaryEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_hidden_layers, n_units_per_layer, activation_function=nn.ReLU()):
        super(Binary2BinaryEncoder, self).__init__()
        layers = [nn.Linear(input_dim, n_units_per_layer), activation_function]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(n_units_per_layer, n_units_per_layer), activation_function])
        layers.extend([nn.Linear(n_units_per_layer, latent_dim), nn.Sigmoid()])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Binary2BinaryDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, n_hidden_layers, n_units_per_layer, activation_function=nn.ReLU()):
        super(Binary2BinaryDecoder, self).__init__()
        layers = [nn.Linear(latent_dim, n_units_per_layer), activation_function]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(n_units_per_layer, n_units_per_layer), activation_function])
        layers.append(nn.Linear(n_units_per_layer, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Binary2BinaryAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, n_hidden_layers, n_units_per_layer,
                 activation_function=nn.ReLU()):
        super(Binary2BinaryAutoencoder, self).__init__()
        self.encoder = Binary2BinaryEncoder(input_dim, latent_dim, n_hidden_layers, n_units_per_layer, activation_function)
        self.decoder = Binary2BinaryDecoder(latent_dim, output_dim, n_hidden_layers, n_units_per_layer, activation_function)

    def forward(self, x, num_fixed=3):
        encoded = self.encoder(x)
        # Use straight-through estimator for training with binary outputs
        encoded_binary = (encoded > 0.5).float().detach() + encoded - encoded.detach()

        # Replace the last 'num_fixed' bits of the encoded output with 0.5
        fixed_values = torch.full((encoded_binary.shape[0], num_fixed), 0.5, device=encoded_binary.device)
        encoded_binary[:, -num_fixed:] = fixed_values.detach()  # Detach to prevent gradients for the fixed part

        decoded = self.decoder(encoded_binary)
        return encoded, decoded


class GoalPredictor(torch.nn.Module):
    def __init__(self, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(n_latent_dims, 1)])
        else:
            self.layers.extend(
                [torch.nn.Linear(n_latent_dims, n_units_per_layer),
                 torch.nn.LeakyReLU(inplace=True), ])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.LeakyReLU(inplace=True), ] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, 1)])

        self.layers.extend([torch.nn.Sigmoid()])
        self.predictor = torch.nn.Sequential(*self.layers)

    def forward(self, z):
        reward = self.predictor(z).squeeze()
        return reward


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

    def forward(self, z0, a, z1):
        a_logits = F.one_hot(a, num_classes=self.n_actions).float()
        context = torch.cat((z0, z1, a_logits), -1)
        reward = self.reward_predictor(context).squeeze()
        return reward


class InvNet(torch.nn.Module):
    def __init__(self, n_actions, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_actions = n_actions

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2 * n_latent_dims, n_actions)])
        else:
            self.layers.extend(
                [torch.nn.Linear(2 * n_latent_dims, n_units_per_layer),
                 torch.nn.Tanh()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.Tanh()] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_actions)])

        self.inv_model = torch.nn.Sequential(*self.layers)

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        a_logits = self.inv_model(context)
        return a_logits


class ContrastiveNet(torch.nn.Module):
    def __init__(self, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.frozen = False

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2 * n_latent_dims, 1)])
        else:
            self.layers.extend(
                [torch.nn.Linear(2 * n_latent_dims, n_units_per_layer),
                 torch.nn.Tanh()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.Tanh()] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, 1)])
        self.layers.extend([torch.nn.Sigmoid()])
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        fakes = self.model(context).squeeze()
        return fakes
