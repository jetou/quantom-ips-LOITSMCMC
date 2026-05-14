import math
import torch

# As an example we'll import an "IDIS" class with methods
# .get_xsections(), .get_x() and .get_q2()
from jamx.idis import IDIS
from quantom_ips.utils.torch_nn_registry import get_dtype


class JAMXTheory:
    def __init__(
        self,
        order: int = 1,
        nx: int = 200,
        xmin: float = 1e-3,
        xmax: float = 0.99,
        nx_input: int = 95,  # 30
        nQ2: int = 200,
        Q20: float = 1.27**2,
        LQ2max: float = 2.5,  # was 4.5
        RS: float = 140.0,
        differential: str = "dxdQ2",
        W2cut: int = 10,
        devices: str = "cpu",
        dtype: str = "float32",
        target: str = "p",
        beampol: float = -1,
        beamhel: float = 0,
    ):
        # Place to initialize any objects needed to obtain x-sections
        # from PDFs
        # Initialize the reaction parameters
        self.target = target
        self.beampol = beampol
        self.beamhel = beamhel
        # Initialize the IDIS object
        self.devices = devices
        self.dtype = get_dtype(dtype)
        # --select order of calculation (0 for LO, 1 for NLO)
        self.order = order  # 1
        # --set up grid on which calculation will be performed
        # --set up x grid
        self.nx = nx  # 100 #--number of points in total
        self.xmin = xmin  # 1e-6 #--lower limit of grid
        self.xmax = xmax  # 0.99 #--upper limit of grid
        self.nx_input = nx_input  # 30
        self.x_input = torch.logspace(
            math.log10(self.xmin),
            math.log10(self.xmax),
            self.nx_input,
            device=self.devices,
            dtype=self.dtype,
        )
        # --set up Q2 grid
        self.nQ2 = nQ2  # 20 #--number of points in total
        self.Q20 = Q20  # 1.27**2 #--input scale and lower limit of grid
        # (1.27**2 is currently the lowest value that works)
        self.LQ2max = LQ2max  # 4.5  #--upper limit of grid (10**LQ2max is taken)
        self.RS = RS  # 140
        self.differential = differential  # "dxdQ2"
        self.W2cut = W2cut  # 10
        temp_input = torch.ones((11, self.nx_input), device=devices, dtype=self.dtype)
        self.idis = IDIS(
            self.devices,
            self.order,
            self.x_input,
            dtype=self.dtype,
            pdf_input=temp_input,
            nx=self.nx,
            xmin=self.xmin,
            xmax=self.xmax,
            nQ2=self.nQ2,
            Q20=self.Q20,
            LQ2max=self.LQ2max,
        )
        return

    def forward(self, pdfs):
        # Input PDFs have shape (batch, type, x, Q2)

        # Should return x-sections, the grid that they should be
        # sampled over, and a loss. x-sections should be a tensor
        # with shape (b, p, x, q), grid is a list with two tensors
        # with shape (x) and (q), and loss is a dictionary as
        # seen below.

        pdfs = torch.mean(pdfs, dim=0, dtype=self.dtype).squeeze()

        self.idis.reset_PDFs(pdfs)
        x_sec = self.idis.get_xsec(
            self.target,
            self.RS,
            self.beampol,
            self.beamhel,
            self.differential,
            self.W2cut,
        )
        x_sec = x_sec[None, None, :, :]
        grid = [self.idis.get_x(), self.idis.get_Q2()]
        loss = {}
        return x_sec, grid, loss
