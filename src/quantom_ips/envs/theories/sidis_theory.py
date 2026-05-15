import sys, os, time
import torch
import numpy as np
from scipy.special import jv, jn_zeros, yv
from . import params, alphaS_
from .lhapdf_setup import configure_lhapdf_paths, get_theory_grids_dir

#--load FFs using LHAPDF
import lhapdf

#--set environment
configure_lhapdf_paths(lhapdf)
lhapdf_set_proton_PDF = 'JAM20-SIDIS_PDF_proton_nlo'
lhapdf_set_pion_FF = 'JAM20-SIDIS_FF_pion_nlo'
PDF_proton = lhapdf.mkPDF(lhapdf_set_proton_PDF,0)
FF_pion = lhapdf.mkPDF(lhapdf_set_pion_FF,0)

flav_dict = {'d':1,'u':2,'s':3,'db':-1,'ub':-2,'sb':-3}

#--set up dictionary for different targets 
PDF = {}
PDF['p'] = lambda X, Q2, f: torch.tensor([PDF_proton.xfxQ2(flav_dict[f],X[i],Q2[i])/X[i] for i in range(len(X))])

#--set up dictionary for different hadron types
FF = {}
FF['pi+'] = lambda Z, Q2, f: torch.tensor([FF_pion.xfxQ2(flav_dict[f],Z[i],Q2[i])/Z[i] for i in range(len(Z))])

def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)

################
#DIMENSION ORDER
################
# (x, Q2, z, bt/qt, phi)
# indicated by einsum indices (x,Q,z,b/q,p)
# B for batch size
# e for ....
# f for flavor

##############
#INTERPOLATORS
##############
#---Interpolation helpers
def local_interp(x0, y0, x):
    
    def get_l(j, x):
        l = 1.0
        for m in range(len(x0)):
            if m == j: continue
            l *= (x - x0[m]) / (x0[j] - x0[m])
        return l
        
    return sum(get_l(j, x) * y0[j] for j in range(len(x0)))

def interp(x0, y0, n, x):
    ix = torch.searchsorted(x0, x)
    ini, fin = ix - n, ix + n + 1
    ini = max(0, ini)
    fin = min(len(x0), fin)
    return local_interp(x0[ini:fin], y0[ini:fin], x)

# General projector (nodes -> eval)
def build_delta(nodes, eval_grid, n=2):
    """
    nodes:     initial points
    eval_grid: final points to interpolation to
    n:         range of location interpolation
    """
    eye = torch.eye(len(nodes))
    return torch.tensor([[interp(nodes, eye[i], n, x) for x in eval_grid] for i in range(len(nodes))])

#############
#THEORY CLASS
#############
class SIDIS:
    '''
    This is the module for 5D SIDIS.  It expects a 4D density as an input with shape (batch, flavs, x, bT).  From this it infers the pixel dimension.
    This module returns the  cross sections, grid_axes (which are needed for samplers like ITS), and a loss that can be optionally used. 

    The output cross section has shape (batch, x, Q2, z, qT, phi)
    '''
    def __init__(
        self,
        order: int = 0,
        nx:    int = 9,
        nQ2:   int = 5,
        nz:    int = 5,
        nqt:   int = 5,
        nbt:   int = 5,
        nphi:  int = 5,
        x_min: float = 0.2,
        x_max: float = 0.8,
        Q2_min: float = 1,
        Q2_max: float = 10.,
        z_min: float = 0.2,
        z_max: float = 0.8,
        bt_min: float = 1e-3,
        bt_max: float = 5.0,
        qt_min: float = 0.2,
        qt_max: float = 1.0,
        phi_min: float = 0.0,
        phi_max: float = 2*torch.pi,
        W2_min:  float = 4,
        tar:     str = 'p',
        had:     str = 'pi+',
        Ebeam:  float = 11,
        average: bool = True,
        log_x:   bool = True,
        log_Q2:  bool = True,
        log_z:   bool = False,
    ):
      
        if order != 0:
            print('Only order==0 supported!  Exiting...')
            sys.exit()
        if tar not in ['p']:
            print('Only tar==p supported!  Exiting...')
            sys.exit()
        if had not in ['pi+']:
            print('Only had==pi+ supported!  Exiting...')
            sys.exit()

        print()
        print()
        print('RUNNING SIDIS THEORY WITH:')
        print('order = %d, tar = %s, had = %s, Ebeam = %3.2f, average = %s'%(order,tar,had,Ebeam,average))
        print('nx = %d, nQ2 = %d, nz = %d, nbt = %d, nqt = %d, nphi = %d'%(nx,nQ2,nz,nbt,nqt,nphi))
        print('x_min = %3.4f, x_max = %3.4f'%(x_min,x_max))
        print('Q2_min = %3.4f, Q2_max = %3.4f'%(Q2_min,Q2_max))
        print('z_min = %3.4f, z_max = %3.4f'%(z_min,z_max))
        print('qt_min = %3.4f, qt_max = %3.4f'%(qt_min,qt_max))
        print('phi_min = %3.4f, phi_max = %3.4f'%(phi_min,phi_max))
        print('W2_min = %3.4f'%(W2_min))
        print('log_x = %s, log_Q2 = %s, log_z = %s'%(log_x,log_Q2,log_z))
        print()
        print()
 
        # Register all the constants here:
        self.order = order
        self.nx = nx
        self.nz = nz
        self.nqt = nqt
        self.nbt = nbt
        self.nQ2 = nQ2
        self.nphi = nphi
        self.x_min = x_min
        self.x_max = x_max
        self.Q2_min = Q2_min
        self.Q2_max = Q2_max
        self.z_min = z_min
        self.z_max = z_max
        self.bt_min = bt_min
        self.bt_max = bt_max
        self.qt_min = qt_min
        self.qt_max = qt_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.W2_min  = W2_min
        self.tar = tar
        self.had = had
        self.Ebeam = Ebeam
        self.average = average
        self.log_x  = log_x
        self.log_Q2 = log_Q2
        self.log_z  = log_z


        #--storage for saving precalculable quantities
        self.storage = {}

        eU2, eD2 = 4/9, 1/9
        self.e2 = {}
        #['d','u','s','sb','ub','db']
        self.e2['p'] = torch.tensor([eD2,eU2,eD2,eD2,eU2,eD2])

        self.nf = len(self.e2)

        #--load other classes
        order = 0
        self.bmax  = 0.8
        ngauss_FT = 50

        if self.log_x: self.x  = torch.logspace(np.log10(self.x_min),np.log10(self.x_max),self.nx)
        else:          self.x  = torch.linspace(self.x_min,self.x_max,self.nx)

        if self.log_Q2: self.Q2 = torch.logspace(np.log10(self.Q2_min),np.log10(self.Q2_max),self.nQ2)
        else:           self.Q2 = torch.linspace(self.Q2_min,self.Q2_max,self.nQ2)
        
        if self.log_z: self.z   = torch.logspace(np.log10(self.z_min),np.log10(self.z_max),self.nz)
        else:          self.z   = torch.linspace(self.z_min,self.z_max,self.nz)

        self.phi = torch.linspace(self.phi_min,self.phi_max,self.nphi)

        #--set up bT
        #self.bt = torch.logspace(np.log10(self.bt_min),np.log10(self.bt_max),self.nbt)
        
        self.nbt_low  = int(np.floor(self.nbt/2))
        self.nbt_high = int(np.ceil(self.nbt/2))
        #--below bmax (log space)
        self.bt_low = torch.logspace(np.log10(self.bt_min),np.log10(self.bmax),self.nbt_low)
        #--above bmax (linear)
        self.bt_high = torch.linspace(self.bmax,self.bt_max,self.nbt_high+1)[1:]
        self.bt = torch.cat((self.bt_low,self.bt_high))

        #---- Non-perturbative model parameters
        #---- gaussian widhts:  PDF   , none  , FF    , GK
        self.prm_f2 = np.array([0.8418, 4.3344, 0.2397, 0.2868])

        #--initialize Fourier Transform
        self.n_interp = 2
        self.build_delta = lambda nodes, eval_grid: build_delta(nodes,eval_grid,n=self.n_interp) 
        self.bTres   = 1000
        self.qtd = torch.linspace(self.qt_min,self.qt_max,self.nqt)
        self.TFT = Tensor_FT(self.qtd,ng=self.nbt, btmin=self.bt_min, btmax=self.bt_max)
        self.bt_eval = torch.linspace(torch.min(self.TFT.bt_1),torch.max(self.TFT.bt_1),self.bTres)
       
        self.build_bTqT_grid()

        #--load Sudakov class
        self.SUDAKOV = MakeSUDAKOV()
       
        #--load OPE class 
        self.OPE = MakeOPE(self.order,self.bmax) 


    def build_bTqT_grid(self):
       
        grid_dir = get_theory_grids_dir()
        filename = 'bTqT_grid_bTres=%d_qTres=%d_ninterp=%d.pt'%(self.nbt,self.nqt,self.n_interp)
        # === Quadrature weights for bT ===
        dbT = torch.diff(self.bt_eval)

        w_bT = torch.empty_like(self.bt_eval)
        w_bT[1:-1] = (dbT[:-1] + dbT[1:]) / 2
        w_bT[0] = dbT[0] / 2
        w_bT[-1] = dbT[-1] / 2

      
        GRIDS = os.listdir(grid_dir)
        if filename in GRIDS:
            self.bTqT_grid = torch.load('%s/%s'%(grid_dir,filename),weights_only=True)#.requires_grad_(True)
            mem = self.bTqT_grid.element_size()*self.bTqT_grid.nelement()/1024**3
            print('Loaded bT -> qT grid from grids/%s (mem = %5.4f GB)'%(filename,mem))
        else:
            t0 = time.time()
            delta_bT = self.build_delta(self.bt, self.bt_eval)  # (nb0, nb_eval)

            # Bessel kernel on eval grid
            J = torch.special.bessel_j0(torch.outer(self.qtd, self.bt_eval))   # (n_q, n_be)
            G = J * (self.bt_eval * w_bT)                                     # (n_q, n_be)
            grid = torch.einsum('be,qe->bqe',delta_bT,G)
            t1 = time.time()
            print('Time taking to generate bT -> qT grid: %3.2f seconds'%(t1-t0))
            print('Saving grid to grids/%s'%filename)
            torch.save(grid,'%s/%s'%(grid_dir,filename))
            self.bTqT_grid = torch.load('%s/%s'%(grid_dir,filename),weights_only=True)#.requires_grad_(True)

    def R_function(self,bt,var, center, width, steepness):
        bt = bt.to(var.device)
        var_2, bt_2 = torch.meshgrid(var,bt,indexing='ij')
        exponent = (bt_2 - center) / width
        return  1- 1.0 / (1 + torch.exp(steepness * exponent))

    def NP_model_PDF(self,bt, x,bmax):
        #
        alpha, _ = self.prm_f2[0:2]
        #
        x = x.to(bt.device)
        center = bmax; width = 1.; steepness = 7.
        R_func   = self.R_function(bt,x, center, width, steepness)
        x_grid, bt_grid = torch.meshgrid(x, bt, indexing='ij')
        pdf = torch.exp(-alpha * bt_grid**2*R_func /4 )
        return pdf

    def NP_model_FF(self, bt, z,bmax):
        #
        alpha = self.prm_f2[2]
        #
        center = bmax; width = 1.; steepness = 7.
        R_func   = self.R_function(bt,z, center, width, steepness) 
        #
        z_grid, bt_grid = torch.meshgrid(z, bt, indexing='ij')
        ff = torch.exp(-alpha * bt_grid**2*R_func / (4 * z_grid**2))
        return ff

    def NP_model_GK(self, bt,Q,bmax):
        '''
        output: ( Q dim, bt dim)
        '''
        center = bmax; width = 1; steepness = 7.
        R_func = self.R_function(bt,Q, center, width, steepness)   
     
        g2 = self.prm_f2[-1]
        #
        Q_0 = Q[0] ##GeV
        Q_2, bt_2 = torch.meshgrid(Q, bt,indexing='ij')
        return torch.exp(-bt_2**2*R_func[0]*g2/2*torch.log(Q_2/Q_0))
    
    def get_sigma0(self,grid):

        x, Q2, z = grid[0], grid[1], grid[2]

        qt = self.qtd.to(x.device).to(x.dtype)

        M = 0.938
        s = 2 * self.Ebeam * M + M**2
        Q = torch.sqrt(Q2)
        shape = Q2.shape
        #_Q2    = Q2.flatten()
        #alpha_em = torch.tensor([self.eweak.get_alpha(q**2) for q in _Q]).reshape(shape).to(self.device)
        alpha_em = 1/137.0
        #y = Q2/x/(s-M**2)
        y = torch.einsum('Q,x->xQ',Q2, 1/x)/(s-M**2)
        #gamma = 2*M*x/Q
        gamma = 2 * M * torch.einsum('x,Q->xQ',x,1/Q)
    
        epsilon = (1 - y - 0.25 * gamma**2 * y**2) / (1 - y + 0.5 * y**2 + 0.25 * gamma**2 * y**2)
    
        #fact1 = 2*torch.pi*alpha_em/x/y**2
        fact1 = 2*torch.pi*alpha_em * torch.einsum('x,xQ->xQ',1/x,1/y**2)

        fact2 = y**2/2/(1-epsilon)
        #fact2 =fact2*(1+gamma**2/2/x)
        fact2 =fact2*(1+ torch.einsum('xQ,x->xQ',gamma**2/2,1/x))
   
        #fact3 = 2*z**2*qt*x
        fact3 = 2*torch.einsum('z,q,x->xzq', z**2, qt, x)

        sigma0 = torch.einsum('xQ,xQ,xzq->xQzq',fact1,fact2,fact3)
        return sigma0

    def forward(self, data):
        t0 = time.time()
        #print('creating grid...')
        grid = self.create_grid(data.shape[2:], data.dtype, data.device)
        #print('getting loss...')
        loss = self.get_loss(data, grid)
        #print('computing cross sections...')
        xsections = self.compute_cross_sections(data, grid)
        #print('cross sections computed')
        t1 = time.time()
        #print(t1-t0)
        return xsections, grid, loss
    
    # This is optional, in case you wish to compute a loss that is somehow unique to the theory your are writing. 
    def get_loss(self, data, grid):
        return {}
   
    # Translate densities (that have been predicted by the generator model) to cross sections, i.e. all your computations go in here:
    def compute_cross_sections(self, data, grid):

        #print('data.shape',data.shape)
        #--data has shape Bfxb

        #--first, evolve TMD PDF (Bfxb -> BfxQb)
        PDF = self.get_tmd_pdf(data,grid)

        #--next, get TMD FF (fixed) (fQzb?)
        FF = self.get_tmd_ff(grid)

        #--then, take product of PDF and FF (BfxQzb)
        product = torch.einsum('BfxQb,fQzb->BfxQzb',PDF,FF)

        #--Fourier transform of PDF
        bTqT_grid = self.bTqT_grid.to(data.device).to(data.dtype)
        FT = torch.einsum('bqe,BfxQzb->BfxQzq',bTqT_grid,product)
 
        #--calculate structure function (flavor sum done here)
        e2 = self.e2[self.tar].to(data.device).to(data.dtype)
        FUUT = torch.einsum('f,BfxQzq->BxQzq',e2,FT)/(2*torch.pi)

        sigma0 = self.get_sigma0(grid)
        sigma = torch.einsum('xQzq,BxQzq->BxQzq',sigma0,FUUT)

        #--add phi dimension (no dependence on it)
        sigma = sigma.unsqueeze(-1).expand(*sigma.shape,self.nphi)
      
        #--ENSURE THAT SIGMA IS POSITIVE
        sigma_pos = torch.where(sigma >= torch.zeros_like(sigma), sigma, 1e-6)

        #--average over batches
        if self.average:
            result = sigma_pos.mean(dim=0, keepdim=True)
            return result
        
        return sigma_pos
   
    def get_tmd_pdf(self,data,grid):
        '''
        model   : bt
        ope     : bt, x, f
        output  : f, x, bt
        '''
        x = grid[0]
        bt = self.bt.to(x.device).to(x.dtype)
        Q2 = grid[1]
        Q  = torch.sqrt(Q2)

        #--multiply data by function that sets it to zero below bmax
        #--bT < bmax: R = 0
        #--bT > bmax: R = 1
        #width, steepness = 0.2, 9.0
        width, steepness = 0.5, 6.0
        R = self.R_function(self.bt,grid[0],self.bmax,width,steepness)

        data = data*R + (1 - R)

        model = data

        if 'sudakov' in self.storage: pert_sdk = self.storage['sudakov']
        else:
            pert_sdk = self.SUDAKOV.perturbative_sudakov(bt,self.bmax,Q)
            self.storage['sudakov'] = pert_sdk

        if 'PDF OPE' in self.storage: ope = self.storage['PDF OPE']
        else:
            ope = self.OPE.get_ope_pdf(self.tar,x,bt)
            self.storage['PDF OPE'] = ope

        if 'gkmod' in self.storage: gkmod = self.storage['gkmod']
        else: 
            gkmod = self.NP_model_GK(bt,Q,self.bmax)
            self.storage['gkmod'] = gkmod

        return torch.einsum('Bfxb,Qb,xbf,Qb->BfxQb',model,pert_sdk,ope,gkmod)

    def get_tmd_ff(self,grid):
        '''
        model   : bt
        ope     : bt, x, f
        output  : f, x, bt
        '''
        if 'TMD FF' in self.storage: return self.storage['TMD FF']
        z = grid[2]
        bt = self.bt.to(z.device).to(z.dtype)
        Q2 = grid[1]
        Q  = torch.sqrt(Q2)
        model = self.NP_model_FF(bt,z,self.bmax)
        pert_sdk = self.SUDAKOV.perturbative_sudakov(bt,self.bmax,Q)
        ope = self.OPE.get_ope_ff(self.had,z,bt) 
        gkmod = self.NP_model_GK(bt,Q,self.bmax)

        tmd_ff = torch.einsum('zb,Qb,zbf,Qb->fQzb',model,pert_sdk,ope,gkmod)
        self.storage['TMD FF'] = tmd_ff
        return tmd_ff        
 
    
    

    
    
    
    # Create a grid that tells LOITS the ranges for the samples observables:
    def create_grid(self,dims, dtype, device):

        #--stuff that will be needed for enforcing kinematic limits
        M = 0.938 #--mass of proton
        M2 = M**2
        Ebeam = self.Ebeam
        s = 2*Ebeam*M + M2

        grid = []
        #--x dimension
        grid.append(self.x.to(device).to(dtype))
        grid.append(self.Q2.to(device).to(dtype))
        grid.append(self.z.to(device).to(dtype))
        grid.append(self.qtd.to(device).to(dtype))
        grid.append(self.phi.to(device).to(dtype))

        return grid


        


class SIDISFreeTMD(SIDIS):
    free_tmd_mode = True  # gan_opt detection

    def __init__(
        self,
        order: int = 0,
        nx:    int = 9,
        nQ2:   int = 5,
        nz:    int = 5,
        nqt:   int = 5,
        nbt:   int = 5,
        nphi:  int = 5,
        x_min: float = 0.2,
        x_max: float = 0.8,
        Q2_min: float = 1,
        Q2_max: float = 10.,
        z_min: float = 0.2,
        z_max: float = 0.8,
        bt_min: float = 1e-3,
        bt_max: float = 5.0,
        qt_min: float = 0.2,
        qt_max: float = 1.0,
        phi_min: float = 0.0,
        phi_max: float = 2*torch.pi,
        W2_min:  float = 4,
        tar:     str = 'p',
        had:     str = 'pi+',
        Ebeam:  float = 11,
        average: bool = True,
        log_x:   bool = True,
        log_Q2:  bool = True,
        log_z:   bool = False,
    ):
        super().__init__(
            order=order,
            nx=nx, nQ2=nQ2, nz=nz, nqt=nqt, nbt=nbt, nphi=nphi,
            x_min=x_min, x_max=x_max,
            Q2_min=Q2_min, Q2_max=Q2_max,
            z_min=z_min, z_max=z_max,
            bt_min=bt_min, bt_max=bt_max,
            qt_min=qt_min, qt_max=qt_max,
            phi_min=phi_min, phi_max=phi_max,
            W2_min=W2_min,
            tar=tar, had=had, Ebeam=Ebeam,
            average=average,
            log_x=log_x, log_Q2=log_Q2, log_z=log_z,
        )

    """
    Variant of SIDIS that expects the generator to output the *full*
    TMD PDF in (x, Q^2, b_T), with no additional OPE/Sudakov evolution.

    Input `data` shape (B, f, nx, nQ2, nbt) or (B, nx, nQ2, nbt).
    Output cross sections: (B, x, Q2, z, qT, phi) as usual.
    """

    def compute_cross_sections(self, data, grid):
        # data: B f x Q b  OR  B x Q b
        if data.ndim == 4:
            # assume (B, x, Q, b) -> add flavor axis
            data = data.unsqueeze(1)

        B, f, nx, nQ2, nbt = data.shape

        # Sanity checks against theory grid
        assert nx  == self.nx,  f"Generator nx={nx} != theory nx={self.nx}"
        assert nQ2 == self.nQ2, f"Generator nQ2={nQ2} != theory nQ2={self.nQ2}"
        assert nbt == self.nbt, f"Generator nbt={nbt} != theory nbt={self.nbt}"

        # Interpret generator output directly as the flavor-resolved TMD PDF
        #   PDF(x, Q, b) = f_gen(x, Q, b)
        PDF = data  # (B, f, x, Q, b)

        # Keep the existing FF model (this still includes its own OPE/Sudakov)
        FF = self.get_tmd_ff(grid)  # (f, Q, z, b)

        # Combine PDF and FF:
        #  PDF: B f x Q b
        #  FF : f Q z b
        # ->  product: B f x Q z b
        product = torch.einsum('BfxQb,fQzb->BfxQzb', PDF, FF)

        # Fourier transform b_T -> q_T using precomputed grid
        bTqT_grid = self.bTqT_grid.to(data.device).to(data.dtype)
        #  bTqT_grid: (b, q, e) with extra "e" index; we sum over b,e
        FT = torch.einsum('bqe,BfxQzb->BfxQzq', bTqT_grid, product)

        # Flavor-summed structure function
        e2 = self.e2[self.tar].to(data.device).to(data.dtype)  # (f,)
        FUUT = torch.einsum('f,BfxQzq->BxQzq', e2, FT) / (2 * torch.pi)

        # Prefactor σ0(x, Q, z, q)
        sigma0 = self.get_sigma0(grid)  # (x, Q, z, q)
        sigma  = torch.einsum('xQzq,BxQzq->BxQzq', sigma0, FUUT)
        #sigma = FUUT  # knockout σ0

        # Add φ dimension (no φ dependence at this level)
        sigma = sigma.unsqueeze(-1).expand(*sigma.shape, self.nphi)

        # Enforce positivity
        sigma_pos = torch.where(sigma >= 0, sigma, torch.full_like(sigma, 1e-6))

        # Average over batches if requested (like original class)
        if self.average:
            return sigma_pos.mean(dim=0, keepdim=True)

        return sigma_pos

    # IMPORTANT: we *do not* override get_tmd_pdf here; we just don't call it.
    # The original SIDIS.get_tmd_pdf is still used elsewhere (e.g. for building
    # the "true" target TMD from density.npy in GANOptimizer).






##########
#OPE class
##########
class MakeOPE:

    def __init__(self,order,bmax):
       
        self.order  = order
        self.bmax   = bmax
        self.flavs  = ['d','u','s','sb','ub','db']
        
        aS     = lambda Q2: alphaS_.get_alphaS(Q2)
        self.aS_vec_np = np.vectorize(aS)

    #---kinematic functions
    def bt_star(self,bt):
        return bt/torch.sqrt(1+bt**2/self.bmax**2)
    
    def mu_b(self,bt):
        C1 = 2*np.exp(-np.euler_gamma)
        return C1/bt

    #---PDF functions 
    def NLO_C_qg_pdf(self,mu2,x):
        fact = torch.tensor (  (self.aS_vec_np(mu2.cpu())*params.TF/2/np.pi) ,device=self.device)
        return fact*(1-x)*2*x 
    
    def NLO_C_qq_pdf(self,mu2,x):
        fact = torch.tensor (  (self.aS_vec_np(mu2.cpu())*params.CF/2/np.pi) ,device=self.device)
        return fact*(1-x)

    def OPE_pdf_nlo_qq(self,rho,x,mu2,flav):
        pdf_1 = torch.tensor(self.pdf_vec(x.cpu()/rho.cpu(),mu2.cpu(),flav),device=self.device)
        return self.NLO_C_qq_pdf(mu2,rho)*pdf_1/rho

    def OPE_pdf_nlo_qg(self,rho,x,mu2,flav):
        pdf_1 = torch.tensor(self.pdf_vec(x.cpu()/rho.cpu(),mu2.cpu(),'g'),device=self.device)
        return self.NLO_C_qg_pdf(mu2,rho)*pdf_1/rho

    def PDF_OPE(self,tar,mu2,x_1,flav):

        order = self.order

        #pdf_function = lambda xx, Q2,flav :self.jamqcd.get_pdf(tar,xx,Q2,flav)
        #self.pdf_vec = np.vectorize(pdf_function)

        #pdf_lo = torch.tensor(self.pdf_vec(x_1.cpu(),mu2.view(-1,1).cpu(),flav),device=self.device)
        #if   order == 0 : return pdf_lo
        #rhos_matrix = []
        #for _ in x_1:
        #    # rho_tmp = torch.linspace(_,1,300,device=self.device)
        #    rho_tmp = torch.logspace(torch.log10(_),0,100,device=self.device)
        #    
        #    rhos_matrix.append(rho_tmp.unsqueeze(1))
        pdf_function  = lambda X, Q2, flav: PDF[tar](X,Q2,flav)

        shape = (x_1.shape[0],mu2.shape[0])
        X, MU2 = torch.meshgrid(x_1,mu2,indexing='ij')
        X   = X.flatten().cpu().numpy()
        MU2 = MU2.flatten().cpu().numpy()

        pdf_lo = pdf_function(X,MU2,flav).reshape(shape).to(x_1.device).to(x_1.dtype)

        if order == 0:
            return pdf_lo
            
        rho_1 = torch.cat(rhos_matrix,dim=1)
        
        int_qq = self.OPE_pdf_nlo_qq(rho_1,x_1,mu2.view(-1,1,1),flav)
        int_qg = self.OPE_pdf_nlo_qg(rho_1,x_1,mu2.view(-1,1,1),flav)
        
        ope_qq = torch.trapz(int_qq,rho_1.unsqueeze(0),dim=1)
        ope_qg = torch.trapz(int_qg,rho_1.unsqueeze(0),dim=1)
        
        if order == 1 : return pdf_lo + ope_qq + ope_qg 

    def get_ope_pdf(self,tar,x_1,bt_1):
        
        btstar_1  = self.bt_star(bt_1) 
        mubstar_1 = self.mu_b(btstar_1)
        ope_pdf = []
        for _ in self.flavs:
            ope_pdf.append(self.PDF_OPE(tar,mubstar_1**2,x_1,_).unsqueeze(2))
        opePDF = torch.cat(ope_pdf,dim=2)
        return opePDF
    
    #---FF functions
    def NLO_C_qg_ff(self,mu2,z):
        fact = torch.tensor (  (self.aS_vec_np(mu2.cpu())*params.CF/2/np.pi) ,device=self.device)
        return fact*( 2*(1+(1-z)**2)*torch.log(z) + z**2 )/z**3

    def NLO_C_qq_ff(self,mu2,z):
        fact = torch.tensor (  (self.aS_vec_np(mu2.cpu())*params.CF/2/np.pi) ,device=self.device)
        return fact*( 2*torch.log(z)*( 2/(1-z) +1/z**2 + 1/z) +1/z**2 - 1/z )

    def OPE_ff_nlo_qq(self,rho,z,mu2,flav):
        pdf_1 = torch.tensor(self.ff_vec(z.cpu()/rho.cpu(),mu2.cpu(),flav),device=self.device)
        return pdf_1*self.NLO_C_qq_ff(mu2,rho)*rho**2/z**2/rho

    def OPE_ff_nlo_qg(self,rho,z,mu2,flav):
        pdf_1 = torch.tensor(self.ff_vec(z.cpu()/rho.cpu(),mu2.cpu(),'g'),device=self.device)
        return pdf_1*self.NLO_C_qg_ff(mu2,rho)*rho**2/z**2/rho
    
    def FF_OPE(self, had,mu2,z_1,flav):
       
        order = self.order 
        ff_function  = lambda Z, Q2, flav: FF[had](Z,Q2,flav)

        shape = (z_1.shape[0],mu2.shape[0])
        Z, MU2 = torch.meshgrid(z_1,mu2,indexing='ij')
        Z   = Z.flatten().cpu().numpy()
        MU2 = MU2.flatten().cpu().numpy()

        ff_lo = ff_function(Z,MU2,flav).reshape(shape).to(z_1.device).to(z_1.dtype)

        if order == 0:
            result = torch.einsum('zQ,z->zQ',ff_lo,1/z_1**2) 
            return result

        #--higher orders not implemented
        rhos_matrix = []
        for _ in z_1:
            # rho_tmp = torch.linspace(_,1,300,device=self.device)
            rho_tmp = torch.logspace(torch.log10(_),np.log(0.999),100,device=self.device)
            
            rhos_matrix.append(rho_tmp.unsqueeze(1))
            
        rho_1 = torch.cat(rhos_matrix,dim=1)
        
        int_qq = self.OPE_ff_nlo_qq(rho_1,z_1,mu2.view(-1,1,1),flav)
        int_qg = self.OPE_ff_nlo_qg(rho_1,z_1,mu2.view(-1,1,1),flav)
        ope_qq = torch.trapz(int_qq,rho_1.unsqueeze(0),dim=1)
        ope_qg = torch.trapz(int_qg,rho_1.unsqueeze(0),dim=1)
        
        if order == 1 : return ff_lo/z_1**2 + ope_qq + ope_qg 

    def get_ope_ff(self, had, z_1,bt_1):
       
        btstar_1  = self.bt_star(bt_1) 
        mubstar_1 = self.mu_b(btstar_1)
        ope_ff = []
        for _ in self.flavs:
            ope_ff.append(self.FF_OPE(had,mubstar_1**2,z_1,_).unsqueeze(2))
        opeFF = torch.cat(ope_ff,dim=2)
        return opeFF


##############
#SUDAKOV CLASS
##############
class MakeSUDAKOV:

    def __init__(self):
        
        aS     = lambda Q2: alphaS_.get_alphaS(Q2)
        self.aS_vec_np = np.vectorize(aS)

    def bt_star(self,bt,bmax):
        return bt/torch.sqrt(1+bt**2/bmax**2)
    
    def mu_b(self,bt):
        C1 = 2*np.exp(-np.euler_gamma)
        return C1/bt

    def gammaK(self,mu):

        nfs_1 = torch.zeros(mu.shape)
        nfs_1[:,:] = 4 
        nfs_1[params.mb<=mu] = 5
        nfs_1[mu<params.mc] = 3
        gammaK_1 = 8*params.CF
        gammaK_2 = params.CA*params.CF*(536/9 - 8*np.pi**2/3) - 80/9*params.CF*nfs_1
        AS =  torch.tensor( self.aS_vec_np(mu.cpu()**2)/4/np.pi)
        return gammaK_1*AS + gammaK_2*AS**2 
        
    def gammaD(self,mu):
        gammaD_1 = 6*params.CF
        AS = torch.tensor( self.aS_vec_np(mu.cpu()**2)/4/np.pi)
        return gammaD_1*AS 

    def perturbative_sudakov(self,bt,bmax,Q):

        btstar = self.bt_star(bt,bmax)
        mu = self.mu_b(btstar).to(bt.device).to(bt.dtype)
        sudakov_integrand = lambda mu,Q:(self.gammaD(mu).to(Q.device).to(Q.dtype) - self.gammaK(mu).to(Q.device).to(Q.dtype)*torch.log(Q/mu))/mu
        qrho_matrix = []
        for qq in Q:
            rhos_matrix = []
            for _ in mu:
                rho_tmp = torch.logspace(torch.log10(_),np.log10(qq.cpu()),150)
                rhos_matrix.append(rho_tmp.unsqueeze(1))
            rho_1   = torch.cat(rhos_matrix,dim=1) 
            qrho_matrix.append(rho_1.unsqueeze(0))

        final_matrix = torch.cat(qrho_matrix,dim=0).to(bt.device).to(bt.dtype)
        sdk_int = sudakov_integrand(final_matrix,Q.view(-1,1,1))
        result     = torch.exp(torch.trapz(sdk_int,final_matrix,dim=1))
        result[torch.isnan(result)]=0
        return result.to(bt.device).to(bt.dtype)



########################
#FOURIER TRANSFORM CLASS
########################
class Tensor_FT:

    def __init__(self,qtd,ng=99,btmin = 1e-3,btmax=10):
    
        self.qtd = qtd
        self.ng = ng
        self.btmin = btmin
        self.btmax = btmax
        
        self.setup()

    def setup(self):
        
        self.qtd_1 = self.qtd
        self.gauss_setup()

    def gauss_setup(self):
        ng = self.ng
        gx,gw = np.polynomial.legendre.leggauss(ng)
        gx_1  = torch.tensor(gx)
        gw_1  = torch.tensor(gw)
        #----
        eta_f = lambda a, b : 0.5*(b-a)*gx_1+0.5*(a+b)
        jac_f = lambda a, b : (b-a)*0.5
        #
        self.eta_1 = eta_f(self.btmin,self.btmax) 
        jac   = jac_f(self.btmin,self.btmax) 
        #---
        qtd_1 = self.qtd_1.to(self.eta_1.dtype)
        eta_2, qtd_2 = torch.meshgrid(self.eta_1,qtd_1,indexing='ij') 
        bessel0_2 = jv(0, qtd_2.cpu()*eta_2.cpu())
        bessel0_2 = bessel0_2
        #---
        self.matr_integral =torch.einsum('g,g,gi->ig',gw_1,self.eta_1,bessel0_2)*jac/2/np.pi
        self.bt_1 =  self.eta_1
        self.bt_2 =  eta_2.T

    def transform(self,function):

        bessel_2  = torch.special.bessel_j0(self.bt_2*self.qt_2)
        FUU_trapz = torch.einsum('bq,...b->...qb',bessel_2,function)
        F_UU      = torch.trapz(FUU_trapz,self.bt_1,dim=-1)
        
        return F_UU






