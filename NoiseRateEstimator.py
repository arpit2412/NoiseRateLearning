import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NoiseRateEstimator:
    """
    Implementation of the EM algorithm for noise rate estimation from the paper:
    "Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation"
    
    This class estimates the noise rate ε and can be integrated with SOTA noisy-label learning methods
    like DivideMix, C2D, InstanceGM, etc.
    """
    
    def __init__(self, feature_dim, num_classes, initial_epsilon=0.5, lambda_val=1.0, lr=0.001):
        """
        Initialize the Noise Rate Estimator.
        
        Args:
            feature_dim (int): Dimension of feature representation
            num_classes (int): Number of classes
            initial_epsilon (float): Initial value for noise rate (default: 0.5)
            lambda_val (float): Weight for the constraint in M-step (default: 1.0)
            lr (float): Learning rate for optimizers (default: 0.001)
        """
        # Initialize noise model parameters
        self.noisy_model = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 256),  # input + one-hot y
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Initialize posterior model parameters
        self.posterior_model = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 256),  # input + one-hot ŷ
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Initialize noise rate parameter (ε)
        self.epsilon_logit = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))  # initialize to logit(0.5) = 0
        
        # Optimizers for each set of parameters
        self.posterior_optimizer = optim.SGD(self.posterior_model.parameters(), lr=lr, momentum=0.9)
        self.noisy_optimizer = optim.SGD(self.noisy_model.parameters(), lr=lr, momentum=0.9)
        self.epsilon_optimizer = optim.SGD([self.epsilon_logit], lr=lr, momentum=0.9)
        
        self.lambda_val = lambda_val
        self.num_classes = num_classes
    
    def get_epsilon(self):
        """Get the current noise rate ε (constrained between 0 and 1)"""
        return torch.sigmoid(self.epsilon_logit).item()
    
    def expectation_step(self, features, y_hat, clean_model):
        """
        E-step: Update the posterior distribution q(y|x,ŷ;ρ)
        
        Args:
            features: Batch of feature representations
            y_hat: Batch of noisy labels
            clean_model: The current clean classifier model (from the integrated method)
        
        Returns:
            q_y: Posterior probabilities for clean labels
        """
        self.posterior_optimizer.zero_grad()
        
        # Forward pass through clean model to get p(y|x;θ_y)
        with torch.no_grad():
            p_y_given_x = F.softmax(clean_model(features), dim=1)
        
        # One-hot encoding of noisy labels
        y_hat_one_hot = F.one_hot(y_hat, num_classes=self.num_classes).float()
        
        # Forward pass through posterior model to get q(y|x,ŷ;ρ)
        posterior_input = torch.cat([features, y_hat_one_hot], dim=1)
        q_y = F.softmax(self.posterior_model(posterior_input), dim=1)
        
        # Compute the variational lower bound (ELBO) in Eq. (2)
        elbo = 0
        epsilon = torch.sigmoid(self.epsilon_logit)
        
        # Loop through all possible clean labels y
        for y_idx in range(self.num_classes):
            # One-hot encoding for current y
            y_one_hot = torch.zeros_like(y_hat_one_hot)
            y_one_hot[:, y_idx] = 1.0
            
            # Get p(y|x;θ_y) for current y
            p_y = p_y_given_x[:, y_idx]
            
            # For p(ŷ|x,y;θ_ŷ,ε)
            noisy_input = torch.cat([features, y_one_hot], dim=1)
            noisy_logits = self.noisy_model(noisy_input)
            p_y_hat_from_model = F.softmax(noisy_logits, dim=1)
            
            # Combine using noise rate: (1-ε)·δ(ŷ=y) + ε·f_θ_ŷ(x,y)
            # When ŷ=y, add (1-ε) probability mass
            same_label_mask = (y_hat == y_idx).float().unsqueeze(1)
            p_y_hat_given_x_y = epsilon * p_y_hat_from_model + (1 - epsilon) * same_label_mask
            
            # Get probability of observed noisy label
            p_y_hat = torch.gather(p_y_hat_given_x_y, 1, y_hat.unsqueeze(1)).squeeze(1)
            
            # log p(ŷ|x,y;θ_ŷ,ε) + log p(y|x;θ_y)
            log_joint = torch.log(p_y_hat + 1e-10) + torch.log(p_y + 1e-10)
            
            # Weight by q(y|x,ŷ;ρ)
            weighted_log_joint = q_y[:, y_idx] * log_joint
            
            elbo += weighted_log_joint.mean()
        
        # Add entropy term: H[q(y|x,ŷ;ρ)]
        entropy = -torch.sum(q_y * torch.log(q_y + 1e-10), dim=1).mean()
        elbo += entropy
        
        # Maximize ELBO (minimize negative ELBO)
        loss = -elbo
        loss.backward()
        self.posterior_optimizer.step()
        
        return q_y.detach()
    
    def maximization_step(self, features, y_hat, q_y, clean_model, clean_indices=None):
        """
        M-step: Update θ_ŷ and ε given the posterior q(y|x,ŷ;ρ)
        
        Args:
            features: Batch of feature representations
            y_hat: Batch of noisy labels
            q_y: Posterior probabilities for clean labels from E-step
            clean_model: The clean classifier model (from the integrated method)
            clean_indices: Indices of samples considered clean based on current ε
            
        Returns:
            loss: The loss value from the M-step
        """
        # Zero gradients for the optimizers
        self.noisy_optimizer.zero_grad()
        self.epsilon_optimizer.zero_grad()
        
        y_hat_one_hot = F.one_hot(y_hat, num_classes=self.num_classes).float()
        
        # Get current p(y|x;θ_y) from clean model (no gradient needed)
        with torch.no_grad():
            p_y_given_x = F.softmax(clean_model(features), dim=1)
        
        # Current noise rate (constrained between 0 and 1)
        epsilon = torch.sigmoid(self.epsilon_logit)
        
        # Compute the variational lower bound (ELBO) for M-step
        elbo = 0
        
        # Loop through all possible clean labels y
        for y_idx in range(self.num_classes):
            # One-hot encoding for current y
            y_one_hot = torch.zeros_like(y_hat_one_hot)
            y_one_hot[:, y_idx] = 1.0
            
            # p(y|x;θ_y) for current y
            p_y = p_y_given_x[:, y_idx]
            
            # For p(ŷ|x,y;θ_ŷ,ε)
            noisy_input = torch.cat([features, y_one_hot], dim=1)
            noisy_logits = self.noisy_model(noisy_input)
            p_y_hat_from_model = F.softmax(noisy_logits, dim=1)
            
            # Combine using noise rate: (1-ε)·δ(ŷ=y) + ε·f_θ_ŷ(x,y)
            same_label_mask = (y_hat == y_idx).float().unsqueeze(1)
            p_y_hat_given_x_y = epsilon * p_y_hat_from_model + (1 - epsilon) * same_label_mask
            
            # Get probability of observed noisy label
            p_y_hat = torch.gather(p_y_hat_given_x_y, 1, y_hat.unsqueeze(1)).squeeze(1)
            
            # log p(ŷ|x,y;θ_ŷ,ε) + log p(y|x;θ_y)
            log_joint = torch.log(p_y_hat + 1e-10) + torch.log(p_y + 1e-10)
            
            # Weight by q(y|x,ŷ;ρ)
            weighted_log_joint = q_y[:, y_idx] * log_joint
            
            elbo += weighted_log_joint.mean()
        
        # No entropy term in M-step since q is fixed
        
        # Loss is negative ELBO (since we're maximizing)
        loss = -elbo
        
        # Backward pass and update parameters
        loss.backward()
        self.noisy_optimizer.step()
        self.epsilon_optimizer.step()
        
        return loss.item(), epsilon.item()
    
    def get_clean_noisy_indices(self, scores, batch_size=None):
        """
        Split data into clean and noisy sets based on estimated noise rate ε
        and provided criterion scores (e.g., loss values)
        
        Args:
            scores: Criterion scores for ranking samples (e.g., loss values)
            batch_size: Optional batch size if different from len(scores)
            
        Returns:
            clean_indices: Indices of samples classified as clean
            noisy_indices: Indices of samples classified as noisy
        """
        # Get current noise rate
        epsilon = self.get_epsilon()
        
        # Define curriculum R(t) = 1 - ε as in Eq. (5)
        clean_ratio = 1.0 - epsilon
        
        # Get actual number of samples
        n_samples = batch_size if batch_size is not None else len(scores)
        
        # Sort samples by scores (ascending order)
        sorted_indices = torch.argsort(scores)
        
        # Select (1-ε) fraction with lowest scores as clean
        num_clean = max(1, int(clean_ratio * n_samples))  # ensure at least 1 clean sample
        clean_indices = sorted_indices[:num_clean]
        noisy_indices = sorted_indices[num_clean:]
        
        return clean_indices, noisy_indices