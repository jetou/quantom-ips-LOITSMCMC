import numpy as np
import torch
import quantom_ips.utils.params as PAR

"""
Collection of functions that allow utilizing
true (or known) pixelated densities (or PDFs)
"""


# Duke & Owens:
duke_and_owens_parameters = [
    1.12995643e-01,
    2.84816039e-02,
    3.49711182e-02,
    -1.07992996e00,
    -8.67724315e-02,
    2.67006979e-02,
    -3.42002556e-01,
    2.16083133e00,
    -7.06215971e-02,
    5.99208979e01,
    -3.80095917e01,
    6.96188257e00,
    -1.93266172e02,
    1.76589359e02,
    -3.96195694e01,
    1.62782187e02,
    -1.79165065e02,
    4.60385729e01,
    -6.26740092e-02,
    3.54363691e-01,
    -1.06902973e-01,
    -1.20394986e00,
    1.55339980e-01,
    -7.87939114e-02,
    2.75190189e-01,
    3.99743001e00,
    -8.40757936e-01,
    4.31272688e01,
    -3.26453695e01,
    7.87648066e00,
    -1.02351414e02,
    1.08446491e02,
    -2.81885597e01,
    2.58442965e01,
    -4.35307825e01,
    1.22755848e01,
]


class DukeAndOwensDensities:
    """
    Returns Duke an Owens Densities.
    User provides the number of points in x and Q2 space.
    """

    def __init__(
        self,
        Q02=1.0,
        lam2=0.4,
        rs=140,
        W2min=10,
        params=duke_and_owens_parameters,
        batch_size=1,
        devices="cpu",
        dtype=torch.float32,
    ):
        self.Q02 = Q02
        self.lam2 = lam2
        self.rs = rs
        self.W2min = W2min
        self.devices = devices
        self.dtype = dtype
        self.batch_size = batch_size

        self.L = torch.as_tensor(
            np.log(self.Q02 / self.lam2), device=self.devices, dtype=self.dtype
        )
        self.parameters = torch.as_tensor(params, dtype=self.dtype, device=devices)[
            None, :
        ]

    def get_s(self, Q2):
        return torch.log(torch.div(torch.log(Q2 / self.lam2), self.L))

    def get_parQ2(self, par, Q2):
        s = self.get_s(Q2)
        return par[0] + par[1] * s + par[2] * s**2

    def get_true_density(self, par, x, Q2):
        A = self.get_parQ2(par[:3], Q2)
        a = self.get_parQ2(par[3:6], Q2)
        b = self.get_parQ2(par[6:9], Q2)
        c = self.get_parQ2(par[9:12], Q2)
        d = self.get_parQ2(par[12:15], Q2)
        e = self.get_parQ2(par[15:18], Q2)

        return (
            A
            * torch.pow(x, a)
            * torch.pow((1 - x), b)
            * (1 + c * x + d * torch.pow(x, 2) + e * torch.pow(x, 3))
        )

    def get_pixelated_density(self, n_points_x, n_points_Q2, params):
        Q2min = PAR.mc2
        Q2max = self.rs**2 - PAR.M2
        LQ2 = torch.exp(
            torch.linspace(
                np.log(Q2min),
                np.log(Q2max),
                n_points_Q2,
                device=self.devices,
                dtype=self.dtype,
            )
        )

        xmin = Q2min / Q2max
        xmax = 1.0
        Lx = torch.exp(
            torch.linspace(
                np.log(xmin),
                np.log(xmax),
                n_points_x,
                device=self.devices,
                dtype=self.dtype,
            )
        )

        xx, yy = torch.meshgrid(Lx, LQ2, indexing="ij")
        return (
            xx,
            yy,
            torch.vmap(lambda p: self.get_true_density(p, xx, yy), in_dims=0)(params),
        )

    def get_pixelated_densities(self, n_points_x, n_points_Q2, to_numpy=False):
        _, _, u_density = self.get_pixelated_density(
            n_points_x, n_points_Q2, self.parameters[:, :18]
        )
        _, _, d_density = self.get_pixelated_density(
            n_points_x, n_points_Q2, self.parameters[:, -18:]
        )

        all_densities = torch.stack([u_density, d_density], dim=1)
        all_densities = torch.repeat_interleave(all_densities, self.batch_size, dim=0)

        if to_numpy:
            return all_densities.detach().cpu().numpy()

        return all_densities


# 2D (and therefore 1D) Proxy Application:
proxy_app_2d_parameters = [0.5, 3.0, 0.3, 4.0, 0.75]


class ProxyApplication2DDensities:

    def __init__(
        self,
        x_min=0.001,
        x_max=0.999,
        y_min=0.001,
        y_max=0.999,
        params=proxy_app_2d_parameters,
        batch_size=1,
        n_targets=1,
        devices="cpu",
        dtype=torch.float32,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.batch_size = batch_size
        self.n_targets = n_targets

        self.devices = devices
        self.dtype = dtype
        self.parameters = torch.as_tensor(params, device=self.devices, dtype=self.dtype)

    def get_true_density(self, p, x, y):
        return (
            torch.pow(x, p[0])
            * torch.pow((1.0 - x), p[1])
            * torch.pow(y, p[2])
            * torch.pow((1.0 - y), p[3])
            * (1.0 + p[4] * x * y)
        )

    def get_pixelated_density(self, n_points_x, n_points_y, to_numpy=False):
        x = torch.linspace(
            self.x_min, self.x_max, n_points_x, device=self.devices, dtype=self.dtype
        )
        y = torch.linspace(
            self.y_min, self.y_max, n_points_y, device=self.devices, dtype=self.dtype
        )

        xx, yy = torch.meshgrid(x, y, indexing="ij")
        density = torch.vmap(lambda par: self.get_true_density(par, xx, yy), in_dims=0)(
            self.parameters[None, :]
        )
        density = density[:, None, :, :]
        density = torch.repeat_interleave(density, self.batch_size, 0)
        density = torch.repeat_interleave(density, self.n_targets, 1)

        if to_numpy:
            return density.detach().cpu().numpy()

        return density
