from opacus import PrivacyEngine
from opacus import privacy_analysis as tf_privacy
import numpy as np
import os
import types
import warnings
from typing import List, Optional, Tuple, Union
import torch

class AdaptivePrivacyEngine(PrivacyEngine):
    def __init__(self, *args, n_accumulation_steps=1, **kwargs):
        super(AdaptivePrivacyEngine, self).__init__(*args, **kwargs)
        self.privacy_ledger = {}
        if 'sample_size' in kwargs:
            self.sample_size = kwargs['sample_size']
        else:
            self.sample_size = args[2]
        self.n_accumulation_steps = n_accumulation_steps

    def update_batch_size(self, new_batch_size, new_n_accumulation_steps):
        self._commit_to_privacy_ledger()
        self.batch_size = new_batch_size
        self.n_accumulation_steps = new_n_accumulation_steps
        self.sample_rate = self.batch_size / self.sample_size

    def update_noise_multiplier(self, new_noise_multiplier):
        self._commit_to_privacy_ledger()
        self.noise_multiplier = new_noise_multiplier

    def _commit_to_privacy_ledger(self):
        privacy_ledger_key = (self.sample_rate, self.noise_multiplier)
        if privacy_ledger_key not in self.privacy_ledger:
            self.privacy_ledger[privacy_ledger_key] = 0
        self.privacy_ledger[privacy_ledger_key] += self.steps

        self.steps = 0

    def get_renyi_divergence(self, sample_rate, noise_multiplier):
        rdp = torch.tensor(
            tf_privacy.compute_rdp(
                sample_rate, noise_multiplier, 1, self.alphas
            )
        )
        return rdp

    def add_query_to_ledger(self, sample_rate, noise_multiplier, n):
        privacy_ledger_key = (sample_rate, noise_multiplier)
        if privacy_ledger_key not in self.privacy_ledger:
            self.privacy_ledger[privacy_ledger_key] = 0
        self.privacy_ledger[privacy_ledger_key] += n

    def get_privacy_spent(
        self, target_delta: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Computes the (epsilon, delta) privacy budget spent so far.
        This method converts from an (alpha, epsilon)-DP guarantee for all alphas that
        the ``PrivacyEngine`` was initialized with. It returns the optimal alpha together
        with the best epsilon.
        Args:
            target_delta: The Target delta. If None, it will default to the privacy
                engine's target delta.
        Returns:
            Pair of epsilon and optimal order alpha.
        """
        if target_delta is None:
            target_delta = self.target_delta

        self._commit_to_privacy_ledger()
        rdp = 0.

        for (sample_rate, noise_multiplier), steps in self.privacy_ledger.items():
            rdp += self.get_renyi_divergence(sample_rate, noise_multiplier) * steps

        return tf_privacy.get_privacy_spent(self.alphas, rdp, target_delta)

class PrivacyFilterEngine(AdaptivePrivacyEngine):
    def __init__(self, epsilon, delta, *args, **kwargs):
        super(PrivacyFilterEngine, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.delta = delta

    def halt(
        self,
        batch_size: Optional[int] = None,
        sample_rate: Optional[float] = None,
        noise_multiplier: Optional[float] = None,
        steps: Optional[int] = 1,
    ) -> bool:
        r"""
        Returns whether the filter would halt if asked to perform one more step
        at the proposed batch_size/sample_rate and noise_multiplier. If None the
        current PrivacyEngine values are used.
        Args:
            batch_size: The proposed new query batch size.
            sample_rate: The proposed new query sample rate (either or both
                batch_size and sample_rate have to be None).
            noise_multiplier: The proposed new query noise multiplier.
            steps: Would the filter halt within this number of steps.
        Returns:
            True (halt) or False (don't halt).
        """
        assert(batch_size is None or sample_rate is None)

        # TODO: implement through a max epsilon for each order alpha, and a
        # direct check of positivity for at least one alpha. Should be much more
        # efficient.

        if batch_size is not None:
            sample_rate = batch_size / self.sample_size
        elif sample_rate is None:
            sample_rate = self.sample_rate

        if noise_multiplier is None:
            noise_multiplier = self.noise_multiplier

        self._commit_to_privacy_ledger()
        self.add_query_to_ledger(sample_rate, noise_multiplier, steps)
        halt = self.get_privacy_spent(target_delta=self.delta)[0] > self.epsilon
        self.add_query_to_ledger(sample_rate, noise_multiplier, -steps)

        return halt

    def step(self):
        if not self.halt():
            super(PrivacyFilterEngine, self).step()

class PrivacyOdometerEngine(AdaptivePrivacyEngine):
    def __init__(
        self,
        delta,
        *args,
        **kwargs,
    ):
        r"""
        Args:
            delta: The target delta for the final (epsion, delta)-DP guarantee.
            *args: Arguments for the underlying PrivacyEngine. See
                https://opacus.ai/api/privacy_engine.html.
            **kwargs: Keyword arguments for the underlying PrivacyEngine.
        """
        super(PrivacyOdometerEngine, self).__init__(*args, **kwargs)

        self.delta = delta
        self.gamma = np.log(2*len(self.alphas)/self.delta) / (np.atleast_1d(self.alphas)-1)

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Computes the (epsilon, delta) privacy budget spent so far.
        This method converts from an (alpha, epsilon)-DP guarantee for all alphas that
        the ``PrivacyEngine`` was initialized with. It returns the optimal alpha together
        with the best epsilon.
        Args:
            target_delta: The Target delta. If None, it will default to the privacy
                engine's target delta.
        Returns:
            Pair of epsilon and optimal order alpha.
        """
        self._commit_to_privacy_ledger()
        rdp = 0.

        for (sample_rate, noise_multiplier), steps in self.privacy_ledger.items():
            rdp += self.get_renyi_divergence(sample_rate, noise_multiplier) * steps

        rdp = torch.max(rdp, self.gamma)
        f = torch.ceil(torch.log2(rdp / self.gamma))
        target_delta = self.target_delta / (len(self.alpjas)*2*torch.pow(f+1, 2))
        rdp = self.gamma * torch.exp2(f)

        return self.get_privacy_spent_heterogeneous_delta(torch.tensor(self.alphas), rdp, target_delta)

    def get_privacy_spent_heterogeneous_delta(
        self, orders: Union[List[float], float], rdp: Union[List[float], float], delta: Union[List[float], float],
    ) -> Tuple[float, float]:
        r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
        multiple RDP orders and target ``delta``.
        Args:
            orders: An array (or a scalar) of orders (alphas).
            rdp: A list (or a scalar) of RDP guarantees.
            delta: A list (or a scalar) of target deltas for each order.
        Returns:
            Pair of epsilon and optimal order alpha.
        Raises:
            ValueError
                If the lengths of ``orders`` and ``rdp`` are not equal.
        """
        orders_vec = np.atleast_1d(orders)
        rdp_vec = np.atleast_1d(rdp)
        delta_vec = np.atleast_1d(delta)

        if len(orders_vec) != len(rdp_vec) or len(orders_vec) != len(delta_vec):
            raise ValueError(
                f"Input lists must have the same length.\n"
                f"\torders_vec = {orders_vec}\n"
                f"\trdp_vec = {rdp_vec}\n"
                f"\tdelta_vec = {delta_vec}\n"
            )

        rdp_vec = torch.tensor(rdp_vec)
        delta = torch.tensor(delta)
        orders_vec = torch.tensor(orders_vec)

        eps = (rdp_vec - torch.log(delta) / (orders_vec - 1)).detach()

        # special case when there is no privacy
        if np.isnan(eps).all():
            return np.inf, np.nan

        idx_opt = np.nanargmin(eps)  # Ignore NaNs
        return eps[idx_opt], orders_vec[idx_opt]

