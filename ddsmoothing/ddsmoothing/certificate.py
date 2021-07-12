import torch
from torch.distributions.normal import Normal
from scipy.stats import norm


class Certificate():
    def compute_proxy_gap(self, logits: torch.Tensor):
        """Compute the proxy gap

        Args:
            logits (torch.Tensor): network outputs
        """
        raise NotImplementedError(
            "base class does not implement this method"
        )

    def sample_noise(self, batch: torch.Tensor, repeated_theta: torch.Tensor):
        """Sample noise to obtain the desired certificate

        Args:
            batch (torch.Tensor): original inputs
            repeated_theta (torch.Tensor): thetas repeated to the same batch
                size as `batch`
        """
        raise NotImplementedError(
            "base class does not implement this method"
        )

    def compute_gap(self, pABar: float):
        """Compute the gap given by this certificate

        Args:
            pABar (float): lower bound of p_A as per the definition of Cohen
                et al
        """
        raise NotImplementedError(
            "base class does not implement this method"
        )

    def compute_radius_estimate(
            self, logits: torch.Tensor, theta: torch.Tensor
    ):
        """Compute a differentiable radius estimate

        Args:
            logits (torch.Tensor): network outputs
            theta (torch.Tensor): parameters of the distribution
        """
        raise NotImplementedError(
            "base class does not implement this method"
        )


class L2Certificate(Certificate):
    norm = "l2"

    def __init__(self, batch_size: int, device: str = "cuda:0"):
        self.m = Normal(
            torch.zeros(batch_size).to(device),
            torch.ones(batch_size).to(device)
        )
        self.device = device

    def compute_proxy_gap(self, logits: torch.Tensor) -> torch.Tensor:
        return self.m.icdf(logits[:, 0].clamp_(0.001, 0.999)) - \
            self.m.icdf(logits[:, 1].clamp_(0.001, 0.999))

    def sample_noise(
            self, batch: torch.Tensor, repeated_theta: torch.Tensor
    ) -> torch.Tensor:
        return torch.randn_like(batch, device=self.device) * repeated_theta

    def compute_gap(self, pABar: float) -> float:
        return norm.ppf(pABar)

    def compute_radius_estimate(
            self, logits: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        return theta/2 * self.compute_proxy_gap(logits)


class L1Certificate(Certificate):
    norm = "l1"

    def __init__(self, device="cuda:0"):
        self.device = device

    def compute_proxy_gap(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, 0] - logits[:, 1]

    def sample_noise(
            self, batch: torch.Tensor, repeated_theta: torch.Tensor
    ) -> torch.Tensor:
        return 2 * (torch.rand_like(batch, device=self.device) - 0.5) * \
            repeated_theta

    def compute_gap(self, pABar: float) -> float:
        return 2 * (pABar - 0.5)

    def compute_radius_estimate(
            self, logits: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        return theta * self.compute_proxy_gap(logits)
