import math 
import jax 
import jax.numpy as jnp
import jax.random as jrn
import jax_cfd.base as cfd
import numpy as np
import torch
from torch import Tensor

from .base import Dynamics


class KolmogorovFlow(Dynamics):
    """Kolmogorov flow dynamics.
    Reference: https://github.com/francois-rozet/sda/

    Args:
        grid_size (int): Size of per edge of the spatial grid.
        reynolds (float): Reynolds number.
        dt (float): Time steps intervals between observations.
        seed (int): RNG seed for jax (to generate initial prior states).
    """

    def __init__(
        self,
        grid_size: int = 128,
        reynolds: float = 1e3,
        dt: float = 0.2,
        seed: int = 42,
        perturb_std: float = 0.0,
    ):
        super().__init__(shape=(2, grid_size, grid_size))
        self.seed = seed
        self.dt = dt 
        self.perturb_std = perturb_std
        grid = cfd.grids.Grid(
            shape=(grid_size, grid_size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),
        )
        bc = cfd.boundaries.periodic_boundary_conditions(2)
        forcing = cfd.forcings.simple_turbulence_forcing(
            grid=grid,
            constant_magnitude=1.0,
            constant_wavenumber=4.0,
            linear_coefficient=-0.1,
            forcing_type="kolmogorov",
        )
        dt_min = cfd.equations.stable_time_step(
            grid=grid,
            max_velocity=5.0,
            max_courant_number=0.5,
            viscosity=1 / reynolds,
        )
        steps = 1 if dt_min > dt else math.ceil(dt / dt_min)
        step_fn = cfd.funcutils.repeated(
            f=cfd.equations.semi_implicit_navier_stokes(
                grid=grid,
                forcing=forcing,
                dt=dt / steps,
                density=1.0,
                viscosity=1 / reynolds,
            ),
            steps=steps,
        )
        self._gens = {}

        
        def _prior(key):
            u, v = cfd.initial_conditions.filtered_velocity_field(
                key,
                grid=grid,
                maximum_velocity=3.0,
                peak_wavenumber=4.0,
            )
            return jnp.stack((u.data, v.data))

        self._prior = jax.jit(jnp.vectorize(_prior, signature="(K)->(C,H,W)"))

        def _transition(uv):
            u, v = cfd.initial_conditions.wrap_variables(
                var=tuple(uv),
                grid=grid,
                bcs=(bc, bc),
            )
            u, v = step_fn((u, v))
            return jnp.stack((u.data, v.data))

        self._transition = jax.jit(jnp.vectorize(_transition, signature="(C,H,W)->(C,H,W)"))
    def _get_gen(self, device) -> torch.Generator:
        if isinstance(device, str):
            device = torch.device(device)

        key = (device.type, device.index)  # ('cpu', None), ('cuda', 0)
        if key not in self._gens:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._gens[key] = gen
        return self._gens[key]

    def prior(self, n_sample):
        key = jrn.PRNGKey(self.seed)
        keys = jrn.split(key, n_sample)
        x = np.array(self._prior(keys))
        return torch.tensor(x)

    def transition(self, x: Tensor) -> Tensor:
        device = x.device
        x = x.detach().cpu().numpy()
        x = np.array(self._transition(x))
        gen = self._get_gen(device)
        noise = torch.randn(x.shape, device = device, generator=gen)
        return torch.tensor(x, device=device) + self.perturb_std * noise * (self.dt ** 0.5)


# import math

# import jax
# import jax.numpy as jnp
# import jax.random as jrn
# import jax_cfd.base as cfd
# import numpy as np
# import torch
# from torch import Tensor

# from .base import Dynamics


# # keep JAX in float32 for speed/memory
# jax.config.update("jax_enable_x64", False)


# class KolmogorovFlow(Dynamics):
#     """Kolmogorov flow dynamics.

#     Optimized points:
#     1. use jax.vmap instead of jnp.vectorize
#     2. keep arrays in float32
#     3. reduce unnecessary conversions
#     4. cache JIT-compiled batched prior/transition
#     """

#     def __init__(
#         self,
#         grid_size: int = 128,
#         reynolds: float = 1e3,
#         dt: float = 0.2,
#         seed: int = 42,
#         max_velocity: float = 5.0,
#         max_courant_number: float = 0.5,
#     ):
#         super().__init__(shape=(2, grid_size, grid_size))

#         self.seed = seed
#         self.dt = dt
#         self.grid_size = grid_size
#         self.reynolds = reynolds

#         grid = cfd.grids.Grid(
#             shape=(grid_size, grid_size),
#             domain=((0, 2 * math.pi), (0, 2 * math.pi)),
#         )
#         bc = cfd.boundaries.periodic_boundary_conditions(2)

#         forcing = cfd.forcings.simple_turbulence_forcing(
#             grid=grid,
#             constant_magnitude=1.0,
#             constant_wavenumber=4.0,
#             linear_coefficient=-0.1,
#             forcing_type="kolmogorov",
#         )

#         dt_min = cfd.equations.stable_time_step(
#             grid=grid,
#             max_velocity=max_velocity,
#             max_courant_number=max_courant_number,
#             viscosity=1.0 / reynolds,
#         )

#         self.inner_steps = 1 if dt_min > dt else math.ceil(dt / dt_min)
#         self.inner_dt = dt / self.inner_steps

#         step_fn = cfd.funcutils.repeated(
#             f=cfd.equations.semi_implicit_navier_stokes(
#                 grid=grid,
#                 forcing=forcing,
#                 dt=self.inner_dt,
#                 density=1.0,
#                 viscosity=1.0 / reynolds,
#             ),
#             steps=self.inner_steps,
#         )

#         def _prior_one(key):
#             u, v = cfd.initial_conditions.filtered_velocity_field(
#                 key,
#                 grid=grid,
#                 maximum_velocity=3.0,
#                 peak_wavenumber=4.0,
#             )
#             return jnp.stack((u.data, v.data)).astype(jnp.float32)

#         def _transition_one(uv):
#             uv = uv.astype(jnp.float32)
#             u, v = cfd.initial_conditions.wrap_variables(
#                 var=(uv[0], uv[1]),
#                 grid=grid,
#                 bcs=(bc, bc),
#             )
#             u, v = step_fn((u, v))
#             return jnp.stack((u.data, v.data)).astype(jnp.float32)

#         # batched, compiled functions
#         self._prior_batched = jax.jit(jax.vmap(_prior_one))
#         self._transition_batched = jax.jit(jax.vmap(_transition_one))

#     def prior(self, n_sample: int) -> Tensor:
#         key = jrn.PRNGKey(self.seed)
#         keys = jrn.split(key, n_sample)
#         x = np.asarray(self._prior_batched(keys), dtype=np.float32)
#         return torch.from_numpy(x)

#     def transition(self, x: Tensor) -> Tensor:
#         device = x.device
#         x_np = np.asarray(x.detach().cpu(), dtype=np.float32)
#         y_np = np.asarray(self._transition_batched(x_np), dtype=np.float32)
#         return torch.from_numpy(y_np).to(device)

#     def __repr__(self):
#         return (
#             f"KolmogorovFlow(grid_size={self.grid_size}, "
#             f"reynolds={self.reynolds}, dt={self.dt}, "
#             f"inner_dt={self.inner_dt:.6f}, inner_steps={self.inner_steps})"
#         )