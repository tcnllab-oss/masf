from __future__ import annotations

from typing import Callable, Iterable, Iterator

import torch
from torch import Tensor


class Dynamics:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def prior(self, n_sample: int) -> Tensor:
        """Return (n_sample, *shape) prior samples on default device."""
        raise NotImplementedError

    def transition(self, x: Tensor) -> Tensor:
        """x: (N, *shape) -> next states with same shape and device."""
        raise NotImplementedError

    def _normalize_batch(self, x: Tensor) -> Tensor:
        """
        Normalize input state to shape (N, *self.shape).
        """
        if x.ndim == len(self.shape):
            x = x.unsqueeze(0)

        if tuple(x.shape[1:]) != self.shape:
            raise ValueError(
                f"Expected state shape {self.shape}, got {tuple(x.shape[1:])}"
            )

        return x

    @torch.no_grad()
    def generate(self, x0: Tensor, steps: int) -> Tensor:
        """
        Generate a full trajectory.

        Args:
            x0: (N, *shape) or (*shape,) initial states.
            steps: number of transitions.

        Returns:
            states: (steps+1, N, *shape)
        """
        if steps < 0:
            raise ValueError("steps must be >= 0")

        x = self._normalize_batch(x0)
        device, dtype = x.device, x.dtype
        N = x.shape[0]

        states = torch.empty((steps + 1, N, *self.shape), device=device, dtype=dtype)
        states[0] = x

        for t in range(steps):
            x = self.transition(x)
            states[t + 1] = x

        return states

    @torch.no_grad()
    def generate_iter(self, x0: Tensor, steps: int) -> Iterator[Tensor]:
        """
        Memory-friendly generator that yields one step at a time.

        Yields:
            x_t with shape (N, *shape), for t = 0..steps.
        """
        if steps < 0:
            raise ValueError("steps must be >= 0")

        x = self._normalize_batch(x0)

        yield x
        for _ in range(steps):
            x = self.transition(x)
            yield x

    @torch.no_grad()
    def state_at(self, x0: Tensor, step: int) -> Tensor:
        """
        Return the state at exactly `step`.

        Args:
            x0: (N, *shape) or (*shape,)
            step: non-negative int

        Returns:
            x_step: (N, *shape)
        """
        if step < 0:
            raise ValueError("step must be >= 0")

        x = self._normalize_batch(x0)

        for _ in range(step):
            x = self.transition(x)

        return x

    @torch.no_grad()
    def states_at(self, x0: Tensor, steps: Iterable[int]) -> Tensor:
        """
        Return states at multiple `steps` efficiently with a single forward sweep.

        Args:
            x0: (N, *shape) or (*shape,)
            steps: iterable of non-negative ints

        Returns:
            xs: (K, N, *shape) where K = len(steps), in the SAME order as given.
        """
        if torch.is_tensor(steps):
            steps_list = [int(s) for s in steps.detach().cpu().tolist()]
        else:
            steps_list = [int(s) for s in steps]

        if not steps_list:
            raise ValueError("steps must be non-empty")
        if any(s < 0 for s in steps_list):
            raise ValueError("steps must be non-negative")

        x = self._normalize_batch(x0)

        uniq_sorted = sorted(set(steps_list))
        out_map: dict[int, Tensor] = {}

        cur_step = 0
        if 0 in uniq_sorted:
            out_map[0] = x.clone()

        max_target = uniq_sorted[-1]
        while cur_step < max_target:
            x = self.transition(x)
            cur_step += 1
            if cur_step in uniq_sorted:
                out_map[cur_step] = x.clone()

        xs = torch.stack([out_map[int(s)] for s in steps_list], dim=0)
        return xs

    @torch.no_grad()
    def iter_states_at(
        self,
        x0: Tensor,
        steps: Iterable[int],
        *,
        return_step: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
        include_step0_progress: bool = False,
    ) -> Iterator[tuple[int, Tensor] | Tensor]:
        """
        Iterate over requested target steps with a single forward sweep.

        Args:
            x0: (N, *shape) or (*shape,)
            steps: iterable of non-negative ints
            return_step: if True, yield (step, state), else yield state
            progress_callback: optional callback(progress_step, max_target_step)
            include_step0_progress: if True, call progress_callback(0, max_target_step) once at start

        Yields:
            (step, x_step) or x_step for requested target steps only.

        Notes:
            - The callback reports INTERNAL trajectory progress for every transition.
            - Example:
                steps=[50]
                callback will receive (1,50), (2,50), ..., (50,50)
                while the iterator itself yields only once at step=50.
        """
        if torch.is_tensor(steps):
            steps_list = [int(s) for s in steps.detach().cpu().tolist()]
        else:
            steps_list = [int(s) for s in steps]

        if not steps_list:
            return
        if any(s < 0 for s in steps_list):
            raise ValueError("steps must be non-negative")

        targets = sorted(set(steps_list))
        max_target = targets[-1]

        x = self._normalize_batch(x0)

        cur = 0

        if progress_callback is not None and include_step0_progress:
            progress_callback(0, max_target)

        for s in targets:
            while cur < s:
                x = self.transition(x)
                cur += 1
                if progress_callback is not None:
                    progress_callback(cur, max_target)

            x_out = x.clone()
            if return_step:
                yield s, x_out
            else:
                yield x_out