from typing import Dict, List, Optional, Union
from pathlib import Path
import torch


# ------------------------------------------------------------
# Loading probing direction
# ------------------------------------------------------------

def load_probe_direction(
    probe_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Load a saved probe_direction from disk.

    Parameters
    ----------
    probe_path : Union[str, Path]
        Path to the probe file.
    device : Optional[torch.device]
        Device to load the tensor onto.
        If None, loads to CPU.

    Returns
    -------
    direction : torch.Tensor
        Unit-norm probing direction δ̂.
        Shape: (B, C, h, w) where B >= 1
        dtype: torch.float32
    """
    probe_path = Path(probe_path)
    
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_path}")
    
    # Load the probe tensor
    if device is None:
        device = torch.device("cpu")
    
    direction = torch.load(probe_path, map_location=device)
    
    # Validate tensor properties
    if not isinstance(direction, torch.Tensor):
        raise ValueError(f"Loaded object is not a torch.Tensor, got {type(direction)}")
    
    if direction.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor (B, C, h, w), got shape {direction.shape}"
        )
    
    if direction.dtype != torch.float32:
        direction = direction.to(torch.float32)
    
    # Ensure tensor is on the specified device
    direction = direction.to(device).detach()
    
    # Normalize to unit norm per sample
    B = direction.shape[0]
    direction_flat = direction.reshape(B, -1)  # (B, C*h*w)
    norms = direction_flat.norm(p=2, dim=1, keepdim=True)  # (B, 1)
    
    if (norms == 0).any():
        raise ValueError("Loaded probe has zero norm, cannot normalize.")
    
    direction_normalized = direction_flat / norms  # (B, C*h*w)
    direction = direction_normalized.reshape(direction.shape)  # (B, C, h, w)
    
    return direction


# ------------------------------------------------------------
# Sampling probing direction
# ------------------------------------------------------------

def sample_unit_direction(
    latent: torch.Tensor,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Sample a unit-norm probing direction with the same shape as the latent.

    Parameters
    ----------
    latent : torch.Tensor
        Reference latent z.
        Shape: (B, C, h, w) where B is batch size
        dtype: torch.float32
    seed : Optional[int]
        Random seed for reproducibility.
        If None, direction is sampled randomly.

    Returns
    -------
    direction : torch.Tensor
        Unit-norm probing direction δ̂.
        Shape: (B, C, h, w)
        Each sample in the batch has unit L2 norm.
        dtype: torch.float32
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Sample from standard normal distribution
    direction = torch.randn_like(latent)

    # Normalize to unit L2 norm per sample
    # Flatten spatial dimensions for each sample
    B = direction.shape[0]
    direction_flat = direction.reshape(B, -1)  # (B, C*h*w)
    
    # Compute L2 norm per sample
    norms = direction_flat.norm(p=2, dim=1, keepdim=True)  # (B, 1)
    
    # Check for zero norms
    if (norms == 0).any():
        raise ValueError("Sampled zero-norm direction, resample required.")
    
    # Normalize each sample
    direction_normalized = direction_flat / norms  # (B, C*h*w)
    # FIX: Use reshape instead of view to handle non-contiguous tensors
    direction = direction_normalized.reshape(latent.shape)  # (B, C, h, w)
    
    return direction


# ------------------------------------------------------------
# Apply probing operator P_epsilon
# ------------------------------------------------------------

def apply_latent_probe(
    latent: torch.Tensor,
    direction: torch.Tensor,
    eps: Union[float, int]
) -> torch.Tensor:
    """
    Apply the latent probing operator P_epsilon(z).

    Parameters
    ----------
    latent : torch.Tensor
        Original latent z.
        Shape: (B, C, h, w)
        dtype: torch.float32
    direction : torch.Tensor
        Unit probing direction δ̂.
        Shape must match latent.
    eps : float or int
        Probing strength ε.

    Returns
    -------
    latent_eps : torch.Tensor
        Probed latent z_ε.
        Shape: (B, C, h, w)
    """
    if latent.shape != direction.shape:
        raise ValueError(
            f"Shape mismatch: latent {latent.shape} vs direction {direction.shape}"
        )

    eps = float(eps)
    latent_eps = latent + eps * direction
    return latent_eps.contiguous()

# ------------------------------------------------------------
# Batch probing for epsilon sweep
# ------------------------------------------------------------

def batch_apply_latent_probe(
    latent: torch.Tensor,
    direction: torch.Tensor,
    eps_list: List[Union[float, int]]
) -> Dict[float, torch.Tensor]:
    """
    Apply probing operator for a list of epsilon values.

    Used for sensitivity-ε curve evaluation.

    Parameters
    ----------
    latent : torch.Tensor
        Original latent z.
        Shape: (B, C, h, w)
    direction : torch.Tensor
        Unit probing direction δ̂.
        Shape: (B, C, h, w)
    eps_list : List[float or int]
        List of probing strengths ε.

    Returns
    -------
    probed_latents : Dict[float, torch.Tensor]
        Mapping from ε to probed latent z_ε.
    """
    probed_latents: Dict[float, torch.Tensor] = {}

    for eps in eps_list:
        eps_float = float(eps)
        probed_latents[eps_float] = apply_latent_probe(
            latent=latent,
            direction=direction,
            eps=eps_float,
        )

    return probed_latents