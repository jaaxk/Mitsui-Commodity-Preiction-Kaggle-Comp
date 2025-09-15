import torch
import torch.nn as nn

class SharpeLoss(nn.Module):
    """
    Custom loss function that optimizes for Sharpe ratio.
    
    The Sharpe ratio is calculated as mean(returns) / std(returns),
    where returns are the differences between predictions and targets.
    
    Args:
        alpha (float, optional): Weighting factor for the standard deviation term.
                               Higher values will penalize volatility more.
                               Default: 1.0
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, preds, targets):
        """
        Calculate the negative Sharpe ratio (to be minimized).
        
        Args:
            preds (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            
        Returns:
            torch.Tensor: Negative Sharpe ratio to be minimized
        """
        # Calculate returns (prediction errors)
        returns = preds - targets
        
        # Calculate mean and standard deviation of returns
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        
        # Negative Sharpe ratio (we want to minimize this)
        sharpe_ratio = -mean_return / (std_return + epsilon)
        
        return sharpe_ratio

