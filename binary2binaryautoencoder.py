import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


mse_loss = nn.MSELoss()
cross_entropy = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss()


# mse loss for auto encoder
def reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return mse_loss(recon_x, x)


# weighted distance loss for pushing neighbour states being similar
def position_weighted_contrastive_loss(z0: torch.Tensor, z1: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    distances = torch.norm(z0 - z1, p=2, dim=2)
    n_features = z0.size(2)
    weights = torch.linspace(2.0, 1.0, steps=n_features).to(z0.device)
    weighted_distances = distances * weights
    weighted_distance = torch.mean(weighted_distances, dim=1)
    loss = torch.mean(labels * torch.pow(weighted_distance, 2))
    return loss


# loss between predicted actions and actual actions from given pairs of state transitions
def inverse_loss(pred_a: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return cross_entropy(input=pred_a, target=a)


# loss for predictions of two states being neighbours or not
def ratio_loss(labels: torch.Tensor, pred_labels: torch.Tensor) -> torch.Tensor:
    return bce_loss(input=pred_labels, target=labels.float())


# loss for predictions of rewards within state-action-state transitions
def reward_loss(pred_r: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return mse_loss(pred_r, r)


def _fix_bits(x: torch.Tensor, num_keep_dim: int) -> torch.Tensor:
    # Replace the last 'num_fixed' bits of the encoded output with 0.5
    fixed_values = torch.full((x.shape[0], num_keep_dim), 0.5, device=x.device)
    x[:, -num_keep_dim:] = fixed_values.detach()  # Detach to prevent gradients for the fixed part
    return x


class Binary2BinaryEncoder(nn.Module):
    def __init__(
            self,
            n_input_dims,
            n_latent_dims,
            n_hidden_layers,
            n_units_per_layer,
            activation_function=nn.LeakyReLU(),
    ):
        super(Binary2BinaryEncoder, self).__init__()
        layers = [nn.Linear(n_input_dims, n_units_per_layer), activation_function]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(n_units_per_layer, n_units_per_layer), activation_function])
        layers.extend([nn.Linear(n_units_per_layer, n_latent_dims), nn.Sigmoid()])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.model(x)
        encoded_binary = (encoded > 0.5).float().detach() + encoded - encoded.detach()
        return encoded_binary


class Binary2BinaryDecoder(nn.Module):
    def __init__(
            self,
            n_latent_dims,
            output_dim,
            n_hidden_layers,
            n_units_per_layer,
            activation_function=nn.LeakyReLU(),
    ):
        super(Binary2BinaryDecoder, self).__init__()
        layers = [nn.Linear(n_latent_dims, n_units_per_layer), activation_function]
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
        self.encoder = Binary2BinaryEncoder(input_dim, latent_dim, n_hidden_layers, n_units_per_layer,
                                            activation_function)
        self.decoder = Binary2BinaryDecoder(latent_dim, output_dim, n_hidden_layers, n_units_per_layer,
                                            activation_function)

    def forward(self, x, num_fixed=3):
        encoded = self.encoder(x)

        # Replace the last 'num_fixed' bits of the encoded output with 0.5
        fixed_values = torch.full((encoded.shape[0], num_fixed), 0.5, device=encoded.device)
        encoded[:, -num_fixed:] = fixed_values.detach()  # Detach to prevent gradients for the fixed part

        decoded = self.decoder(encoded)
        return encoded, decoded


class TerminationPredictor(torch.nn.Module):
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


class Binary2BinaryFeatureNet(torch.nn.Module):
    def __init__(
            self,
            n_actions: int,
            n_obs_dims: int,
            n_latent_dims=32,
            lr=0.001,
            weights=None,
            device=torch.device('cpu')
    ):
        super().__init__()
        if weights is None:
            weights = {'inv': 0.2, 'dis': 0.2, 'neigh': 0.2, 'dec': 0.2, 'rwd': 0.2, 'terminate': 0.2}
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.device = device
        self.weights = weights

        self.encoder = Binary2BinaryEncoder(
            n_input_dims=n_obs_dims,
            n_latent_dims=n_latent_dims,
            n_hidden_layers=3,
            n_units_per_layer=256,
        ).to(device)

        if weights['inv'] > 0.0:
            self.inv_model = InvNet(
                n_actions=n_actions,
                n_latent_dims=n_latent_dims,
                n_units_per_layer=3,
                n_hidden_layers=128,
            ).to(device)
        else:
            self.inv_model = None

        if weights['dis'] > 0.0:
            self.discriminator = ContrastiveNet(
                n_latent_dims=n_latent_dims,
                n_hidden_layers=3,
                n_units_per_layer=128,
            ).to(device)
        else:
            self.discriminator = None

        if weights['dec'] > 0.0:
            self.decoder = Binary2BinaryDecoder(
                n_latent_dims=n_latent_dims,
                output_dim=n_obs_dims,
                n_hidden_layers=3,
                n_units_per_layer=256,
            ).to(device)
        else:
            self.decoder = None

        if weights['rwd'] > 0.0:
            self.reward_predictor = RewardPredictor(
                n_actions=n_actions,
                n_latent_dims=n_latent_dims,
                n_hidden_layers=3,
                n_units_per_layer=128,
            ).to(device)
        else:
            self.reward_predictor = None

        if weights['terminate'] > 0.0:
            self.termination_predictor = TerminationPredictor(
                n_latent_dims=n_latent_dims,
                n_hidden_layers=1,
                n_units_per_layer=128,
            ).to(device)
        else:
            self.reward_predictor = None

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.cross_entropy = torch.nn.CrossEntropyLoss().to(device)
        self.bce_loss = torch.nn.BCELoss().to(device)
        self.mse = torch.nn.MSELoss().to(device)

    def forward(self, obs_vec):
        raise NotImplementedError

    def compute_loss(self, x0, x1, z0, z1, a, r=None, d=None):
        loss = torch.tensor(0.0).to(self.device)
        inv_loss = self.inverse_loss(z0, z1, a) if self.weights['inv'] > 0.0 else torch.tensor(0.0)
        ratio_loss = self.ratio_loss(z0, z1) if self.weights['dis'] > 0.0 else torch.tensor(0.0)
        neighbour_loss = self.mse(z0, z1) if self.weights['neighbour'] > 0.0 else torch.tensor(0.0)
        pixel_loss = 0.5 * (self.pixel_loss(x0, z0) + self.pixel_loss(x1, z1)) if self.weights[
                                                                                      'dec'] > 0.0 else torch.tensor(
            0.0)
        loss += self.weights['inv'] * inv_loss
        loss += self.weights['dis'] * ratio_loss
        loss += self.weights['neighbour'] * neighbour_loss
        loss += self.weights['dec'] * pixel_loss

        # Initialize reward_loss and demo_loss to zero
        reward_loss = torch.tensor(0.0)
        demo_loss = torch.tensor(0.0)

        # Compute reward_loss if 'r' is not None and add it to the total loss
        if r is not None:
            reward_loss = self.reward_loss(z0, z1, a, r) if self.weights['rwd'] > 0.0 else torch.tensor(0.0)
            loss += self.weights['rwd'] * reward_loss

        # Compute demo_loss if 'd' is not None and add it to the total loss
        if d is not None:
            demo_loss = self.demo_loss(z0, z1, d) if self.weights['demo'] > 0.0 else torch.tensor(0.0)
            loss += self.weights['demo'] * demo_loss

        # Return the total loss and the individual components
        return loss, inv_loss, neighbour_loss, ratio_loss, pixel_loss, reward_loss, demo_loss

    def train_batch(
            self,
            obs_vec0: torch.Tensor,
            actions: torch.Tensor,
            obs_vec1: torch.Tensor,
            rewards: torch.Tensor,
            is_terminated: torch.Tensor,
            num_keep_dim: int,
    ):
        assert len(obs_vec0) == len(actions) == len(obs_vec1) == len(rewards) == len(is_terminated), "input dimension mismatch"
        assert len(obs_vec0) >= 2, "at least more than 2 samples"

        self.train()
        self.phi.train()
        self.optimizer.zero_grad()

        # encode obs 0, obs 1
        z0 = self.encoder(obs_vec0)
        z0 = _fix_bits(z0, num_keep_dim)
        z1 = self.encoder(obs_vec1)
        z1 = _fix_bits(z1, num_keep_dim)

        # get fake z1
        idx = torch.randperm(len(obs_vec1))
        fake_z1 = z1.view(len(z1), -1)[idx].view(z1.size())

        # compute reconstruct loss
        decoded_z0 = self.decoder(z0)
        decoded_z1 = self.decoder(z1)
        rec_loss = self.mse(torch.cat((decoded_z0, decoded_z1), dim=0), torch.cat((obs_vec0, obs_vec1), dim=0))

        # compute inverse loss
        pred_actions = self.inv_model(z0, z1)
        inv_loss = self.cross_entropy(pred_actions, actions)

        # compute ratio loss
        # real transitions = 1s; fake transitions = 0s
        labels = torch.cat((
            torch.ones(len(z1), device=z1.device),
            torch.zeros(len(fake_z1), device=fake_z1.device),
        ), dim=0)
        pred_fakes = torch.cat((
            self.discriminator(z0, z1),
            self.discriminator(z0, fake_z1),
        ), dim=0)
        ratio_loss = self.bce_loss(pred_fakes, labels)

        # compute terminate loss
        pred_terminated = self.termination_predictor(z1)
        terminate_loss = self.bce_loss(pred_terminated, is_terminated)

        # compute neighbour loss
        distances = torch.norm(z0 - z1, p=2, dim=2)
        weights = torch.linspace(1.0, 2.0, steps=z0.size(2)).to(z0.device)
        weighted_distances = distances * weights
        weighted_distance = torch.mean(weighted_distances, dim=1)
        neighbour_loss = torch.mean(labels * torch.pow(weighted_distance, 2))

        z0 = self.phi(x0)
        z1 = self.phi(x1)
        # z1_hat = self.fwd_model(z0, a)
        loss, inv_loss, neighbour_loss, ratio_loss, pixel_loss, reward_loss, demo_loss = self.compute_loss(x0, x1, z0,
                                                                                                           z1, a, r, d)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item(), inv_loss.detach().cpu().item(), neighbour_loss.detach().cpu().item(), ratio_loss.detach().cpu().item(), pixel_loss.detach().cpu().item(), reward_loss.detach().cpu().item(), demo_loss.detach().cpu().item()

    def save(self, checkpoint_path, counter=-1, _counter=-1, performance=0.0):
        torch.save(
            {
                'counter': counter,
                '_counter': _counter,
                'encoder': self.encoder.state_dict(),
                'inv_model': self.inv_model.state_dict() if self.weights['inv'] > 0.0 else None,
                'discriminator': self.discriminator.state_dict() if self.weights['dis'] > 0.0 else None,
                'decoder': self.decoder.state_dict() if self.weights['dec'] > 0.0 else None,
                'reward_predictor': self.reward_predictor.state_dict() if self.weights['rwd'] > 0.0 else None,
                'termination_predictor': self.termination_predictor.state_dict() if self.weights['rwd'] > 0.0 else None,
                'optimizer': self.optimizer.state_dict(),
                'performance': performance,
                'weights': self.weights,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        weights = checkpoint['weights']  # not self
        self.encoder.load_state_dict(checkpoint['encoder'])
        if weights['inv'] > 0.0:
            self.inv_model.load_state_dict(checkpoint['inv_model'])
        if weights['dis'] > 0.0:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
        if weights['dec'] > 0.0:
            self.decoder.load_state_dict(checkpoint['decoder'])
        if weights['rwd'] > 0.0:
            self.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        if weights['terminate'] > 0.0:
            self.termination_predictor.load_state_dict(checkpoint['termination_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['counter'], checkpoint['_counter'], checkpoint['performance']

class Binary2BinaryTrainer:
    def __init__(self, model_dir: str, model_name: str, device):
        self.device = device
