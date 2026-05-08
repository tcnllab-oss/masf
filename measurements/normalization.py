import torch


def normalizer(center, scale, eps=1e-5):
    return lambda x: (x - center) / (scale + eps)


def denormalizer(center, scale, eps=1e-5):
    return lambda x_n: x_n * (scale + eps) + center


def normalization(prior, device, norm_type="standard", channel_dim=1, eps=1e-5):
    """
    Compute per-channel normalization stats for tensors shaped [B, C, ...].

    Returns:
        center, scale with broadcastable shape [1, C, 1, ...]
    """
    x = prior.detach().to(device)

    if channel_dim != 1:
        dims = list(range(x.ndim))
        dims[1], dims[channel_dim] = dims[channel_dim], dims[1]
        x = x.permute(*dims)

    reduce_dims = tuple(d for d in range(x.ndim) if d != 1)

    if norm_type == "robust":
        # sequential median over non-channel dims
        center = x
        for d in sorted(reduce_dims, reverse=True):
            center = center.median(dim=d, keepdim=True).values

        abs_dev = (x - center).abs()
        scale = abs_dev
        for d in sorted(reduce_dims, reverse=True):
            scale = scale.median(dim=d, keepdim=True).values

        scale = (1.4826 * scale).clamp_min(eps)

    elif norm_type == "standard":
        center = x.mean(dim=reduce_dims, keepdim=True)
        scale = x.std(dim=reduce_dims, keepdim=True).clamp_min(eps)

    elif norm_type == "minmax":
        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
        center = 0.5 * (x_max + x_min)
        scale = 0.5 * (x_max - x_min)
        scale = scale.clamp_min(eps)

    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    return center.to(device), scale.to(device)


class NormalizationManager:
    def __init__(
        self,
        device="cuda",
        eps=1e-5,
        same_normalization=True,
        normalization_form="affine",      # "affine" | "scale_only"
        stats_mode="standard",            # "standard" | "robust" | "minmax"
        stats_update_mode="fixed",        # None | "fixed" | "adaptive" | "moving"
        momentum=0.05,                    # used only for moving
    ):
        self.device = torch.device(device)
        self.eps = eps

        self.same_normalization = same_normalization
        self.normalization_form = normalization_form
        self.stats_mode = stats_mode
        self.stats_update_mode = stats_update_mode
        self.momentum = momentum

        # per-channel stats
        self.u = None
        self.s = None
        self.u_z = None
        self.s_z = None

        # scalar scales for scale_only mode
        self.data_std_x = None
        self.data_std_z = None

        self.initialized = False

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    def _stats(self, x):
        return normalization(
            x,
            self.device,
            norm_type=self.stats_mode,
            eps=self.eps,
        )

    def _scalar_scale(self, scale_tensor):
        if not torch.is_tensor(scale_tensor):
            scale_tensor = torch.tensor(float(scale_tensor), device=self.device)
        return torch.clamp(scale_tensor.mean(), min=self.eps)

    def _ema_update(self, old, new):
        if old is None:
            return new
        return (1.0 - self.momentum) * old + self.momentum * new

    def _check_affine_stats(self, is_z=False):
        if is_z:
            if self.same_normalization:
                if self.u is None or self.s is None:
                    raise RuntimeError("X stats are not initialized. Call update_stats(...).")
            else:
                if self.u_z is None or self.s_z is None:
                    raise RuntimeError("Z stats are not initialized. Call update_stats(...).")
        else:
            if self.u is None or self.s is None:
                raise RuntimeError("X stats are not initialized. Call update_stats(...).")

    def _check_scale_only_stats(self, is_z=False):
        if is_z:
            if self.same_normalization:
                if self.data_std_x is None:
                    raise RuntimeError("X scale stats are not initialized. Call update_stats(...).")
            else:
                if self.data_std_z is None:
                    raise RuntimeError("Z scale stats are not initialized. Call update_stats(...).")
        else:
            if self.data_std_x is None:
                raise RuntimeError("X scale stats are not initialized. Call update_stats(...).")

    def _normalize_affine(self, x, center, scale):
        return (x - center) / (scale + self.eps)

    def _denormalize_affine(self, x, center, scale):
        return x * (scale + self.eps) + center

    def _normalize_scale_only(self, x, data_std):
        return x / (data_std + self.eps)

    def _denormalize_scale_only(self, x, data_std):
        return x * (data_std + self.eps)

    def _compute_new_stats(self, prior, H_phy=None):
        prior = prior.to(self.device)

        need_z = (H_phy is not None)
        z_prior = H_phy(prior) if need_z else None

        new_u, new_s = self._stats(prior)

        if self.same_normalization:
            new_u_z, new_s_z = new_u, new_s
        else:
            if z_prior is None:
                raise ValueError("H_phy must be provided when same_normalization=False.")
            new_u_z, new_s_z = self._stats(z_prior)

        new_data_std_x = self._scalar_scale(new_s)
        new_data_std_z = new_data_std_x if self.same_normalization else self._scalar_scale(new_s_z)

        return new_u, new_s, new_u_z, new_s_z, new_data_std_x, new_data_std_z

    # -------------------------------------------------
    # main API
    # -------------------------------------------------
    def update_stats(self, prior, H_phy=None):
        """
        stats_mode:
            - standard
            - robust
            - minmax

        normalization_form:
            - affine      : (x - center) / scale
            - scale_only  : x / scale

        stats_update_mode:
            - None       : identity normalization
            - fixed      : initialize once, then keep forever
            - adaptive   : recompute every call
            - moving     : EMA update
        """
        if self.stats_update_mode is None:
            self.initialized = True
            return

        new_u, new_s, new_u_z, new_s_z, new_data_std_x, new_data_std_z = self._compute_new_stats(
            prior, H_phy=H_phy
        )

        if self.stats_update_mode == "fixed":
            if self.initialized:
                return

            self.u, self.s = new_u, new_s
            self.u_z, self.s_z = new_u_z, new_s_z
            self.data_std_x = new_data_std_x
            self.data_std_z = new_data_std_z
            self.initialized = True
            return

        if self.stats_update_mode == "adaptive":
            self.u, self.s = new_u, new_s
            self.u_z, self.s_z = new_u_z, new_s_z
            self.data_std_x = new_data_std_x
            self.data_std_z = new_data_std_z
            self.initialized = True
            return

        if self.stats_update_mode == "moving":
            self.u = self._ema_update(self.u, new_u)
            self.s = self._ema_update(self.s, new_s)
            self.u_z = self._ema_update(self.u_z, new_u_z)
            self.s_z = self._ema_update(self.s_z, new_s_z)
            self.data_std_x = self._ema_update(self.data_std_x, new_data_std_x)
            self.data_std_z = self._ema_update(self.data_std_z, new_data_std_z)
            self.initialized = True
            return

        raise ValueError(f"Unknown stats_update_mode: {self.stats_update_mode}")

    # -------------------------------------------------
    # x normalization
    # -------------------------------------------------
    def norm(self, x):
        if self.stats_update_mode is None:
            return x

        if self.normalization_form == "scale_only":
            self._check_scale_only_stats(is_z=False)
            return self._normalize_scale_only(x, self.data_std_x)

        if self.normalization_form == "affine":
            self._check_affine_stats(is_z=False)
            return self._normalize_affine(x, self.u, self.s)

        raise ValueError(f"Unknown normalization_form: {self.normalization_form}")

    def denorm(self, x):
        if self.stats_update_mode is None:
            return x

        if self.normalization_form == "scale_only":
            self._check_scale_only_stats(is_z=False)
            return self._denormalize_scale_only(x, self.data_std_x)

        if self.normalization_form == "affine":
            self._check_affine_stats(is_z=False)
            return self._denormalize_affine(x, self.u, self.s)

        raise ValueError(f"Unknown normalization_form: {self.normalization_form}")

    # -------------------------------------------------
    # z normalization
    # -------------------------------------------------
    def norm_z(self, x):
        if self.stats_update_mode is None:
            return x

        if self.same_normalization:
            return self.norm(x)

        if self.normalization_form == "scale_only":
            self._check_scale_only_stats(is_z=True)
            return self._normalize_scale_only(x, self.data_std_z)

        if self.normalization_form == "affine":
            self._check_affine_stats(is_z=True)
            return self._normalize_affine(x, self.u_z, self.s_z)

        raise ValueError(f"Unknown normalization_form: {self.normalization_form}")

    def denorm_z(self, x):
        if self.stats_update_mode is None:
            return x

        if self.same_normalization:
            return self.denorm(x)

        if self.normalization_form == "scale_only":
            self._check_scale_only_stats(is_z=True)
            return self._denormalize_scale_only(x, self.data_std_z)

        if self.normalization_form == "affine":
            self._check_affine_stats(is_z=True)
            return self._denormalize_affine(x, self.u_z, self.s_z)

        raise ValueError(f"Unknown normalization_form: {self.normalization_form}")

    def sigma_norm_z(self, noise_std):
        if self.stats_update_mode is None:
            return noise_std

        if self.normalization_form == "scale_only":
            self._check_scale_only_stats(is_z=True)
            data_std = self.data_std_x if self.same_normalization else self.data_std_z
            return noise_std / (data_std + self.eps)

        if self.normalization_form == "affine":
            if self.same_normalization:
                self._check_affine_stats(is_z=False)
                return noise_std / (self.s + self.eps)
            else:
                self._check_affine_stats(is_z=True)
                return noise_std / (self.s_z + self.eps)

        raise ValueError(f"Unknown normalization_form: {self.normalization_form}")