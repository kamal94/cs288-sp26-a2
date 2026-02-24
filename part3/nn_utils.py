"""
Neural network utilities for Transformer implementation.
Contains basic building blocks: softmax, cross-entropy, gradient clipping, token accuracy, perplexity.
"""
from logging import logProcesses
import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    # TODO: Implement numerically stable softmax. You can re-use the same one 
    # used in part 2. But for this problem, you need to implement a numerically stable version to pass harder tests.
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

    raise NotImplementedError("Implement softmax")


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Unnormalized log probabilities of shape (N, C) where N is batch size
                and C is number of classes
        targets: Ground truth class indices of shape (N,)
    
    Returns:
        Scalar tensor containing the mean cross-entropy loss
    """
    # TODO: Implement cross-entropy loss
    N = targets.shape[0]
    normalized = softmax(logits, dim=-1)
    one_hot_vectors = torch.zeros(normalized.shape)
    for i, entry in enumerate(targets):
        one_hot_vectors[i][entry] = 1
    # inline softmax for precision
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_probabilities = shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
    return - torch.sum(one_hot_vectors * log_probabilities)/N
    raise NotImplementedError("Implement cross_entropy")


def gradient_clipping(parameters, max_norm: float) -> Tensor:
    """
    Clip gradients of parameters by global norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns:
        The total norm of the gradients before clipping
    """
    # TODO: Implement gradient clipping
    # for parameter in parameters:
    total_norm = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total_norm += torch.norm(parameter.grad) ** 2

    total_norm = torch.sqrt(total_norm)

    if total_norm > max_norm:
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter.grad.data *= max_norm / total_norm
    return total_norm


def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute token-level accuracy for language modeling.
    
    Computes the fraction of tokens where the predicted token (argmax of logits)
    matches the target token, ignoring positions where target equals ignore_index.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing accuracy (default: -100)
    
    Returns:
        Scalar tensor containing the accuracy (between 0 and 1)
    
    Example:
        >>> logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> token_accuracy(logits, targets)
        tensor(1.)  # All predictions correct: argmax gives [0, 1, 2]
        
        >>> logits = torch.tensor([[2.0, 1.0], [0.1, 3.0], [1.0, 0.5]])
        >>> targets = torch.tensor([1, 1, 0])
        >>> token_accuracy(logits, targets)
        tensor(0.6667)  # 2 out of 3 correct
    """
    # TODO: Implement token accuracy
    total_correct = 0
    ignored = 0
    N = targets.shape[0]
    for i in range(N):
        if targets[i] == ignore_index:
            ignored += 1
            continue
        high = -float("inf")
        high_index = -1
        for j in range(logits.shape[-1]):
            if logits[i][j] > high:
                high = logits[i][j]
                high_index = j
        if high_index == targets[i]:
            total_correct += 1
    print(f"{ignored=}")
    print(f"{N=}")
    print(f"{total_correct=}")
    return torch.tensor(total_correct/(N-ignored))
            

    raise NotImplementedError("Implement token_accuracy")


def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute perplexity for language modeling.
    
    Perplexity is defined as exp(cross_entropy_loss). It measures how well the
    probability distribution predicted by the model matches the actual distribution
    of the tokens. Lower perplexity indicates better prediction.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing perplexity (default: -100)
    
    Returns:
        Scalar tensor containing the perplexity (always >= 1)
    
    Example:
        >>> # Perfect predictions (one-hot logits matching targets)
        >>> logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(1.0001)  # Close to 1 (perfect)
        
        >>> # Uniform predictions (high uncertainty)
        >>> logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(3.)  # Equal to vocab_size (worst case for uniform)
    """
    # TODO: Implement perplexity
    
    raise NotImplementedError("Implement perplexity")
