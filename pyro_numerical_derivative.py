import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
#from torch.distributions import half_normal, normal 

from numpy import random
rs = random.RandomState(0)
import pandas as pd
#from distributed_model_flows import init_flow_model

import matplotlib.pyplot as plt
import matplotlib as mpl 
from scipy.optimize import minimize

from obj import PyTorchObjective
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from hessian import hessian as hessianrepo 
import pickle as pkl
import pyro
from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform, LogNormal, HalfNormal, MultivariateNormal
from pyro.distributions.util import scalar_like
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import initialize_model, summary
from pyro.util import ignore_experimental_warning

from torch.multiprocessing import Pool, set_start_method, freeze_support

from torch.distributions import constraints, normal, half_normal, log_normal, uniform 
from torch.distributions.utils import broadcast_all
from pyro.distributions import TorchDistribution

#import arviz as az
#import make_cmdlogp_plot

from scipy.stats import norm

import corner
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular
#from nflows.flows.base import Flow
#from nflows.distributions.normal import StandardNormal
#from nflows.transforms.base import CompositeTransform
#from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
#from nflows.transforms.permutations import ReversePermutation


def gen_model(scale=10, fraction=0.5):
    #build normalizing flow model from previous fit
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    args = pkl.load(open('args.pkl', 'rb'))
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, 5, regularization_fns).to(device)#.cuda()
    if args.spectral_norm: add_spectral_norm(model)
    set_cnf_options(args, model)
    model.load_state_dict( torch.load('model_10000.pt'))
 
    #if torch.cuda.is_available():
    #    model = init_flow_model(
    #        num_inputs=5,
    #        num_cond_inputs=None).cuda() #len(cond_cols)).cuda()
    #else:
    #    model = init_flow_model(
    #        num_inputs=5,
    #        num_cond_inputs=None) #len(cond_cols)).cuda()
    
    #num_layers = 5
    #base_dist = StandardNormal(shape=(5,))
    #transforms = []
    #for _ in range(num_layers):
    #    transforms.append(ReversePermutation(features=5))
    #    transforms.append(MaskedAffineAutoregressiveTransform(features=5, 
    #                                                      hidden_features=4))
    #transform = CompositeTransform(transforms)
    #model = Flow(transform, base_dist).to(device)

    #model.cpu()
    #filename = 'checkpoint11434epochs_cycle.pth'
    #filename = f'gauss_scale{scale}_frac{fraction}/checkpoint200000epochs_cycle_gauss.pth'
    #filename = 'gauss_scale10_frac0.25/checkpoint100000epochs_cycle_gauss.pth'
    #filename = 'checkpoint_epoch{}.pth'.format(95000)
    #data = torch.load(filename, map_location=device)
    #breakpoint()
    #model.load_state_dict(data['model'])
    #if torch.cuda.is_available():
    #    data = torch.load(filename)
    #    model.load_state_dict(data['model'])
    #    model.cuda();
    #else:
    #    data = torch.load(filename, map_location=torch.device('cpu'))
    #    model.load_state_dict(data['model'])
    return model



class Objective(nn.Module):
    def __init__(self, chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss):
        super(Objective, self).__init__()
        #self.lnA = nn.Parameter(torch.Tensor([theta[0]]))
        #self.M = nn.Parameter(torch.Tensor([theta[1]]))
        #self.c = nn.Parameter(torch.Tensor([theta[2]]))
        #self.lnd = nn.Parameter(torch.Tensor([theta[3]]))
        self.chat = chat
        self.mhat = mhat
        self.varpihat = varpihat
        self.sigmac = sigmac
        self.sigmam = sigmam
        self.sigmavarpi = sigmavarpi
        self.dustco_c = dustco_c
        self.dustco_m = dustco_m
        self.ss = ss

    def forward(self):
        return -1*self.logjoint()
    
    def predicted_color(self, A, c, coeff, pyt=True):
        if pyt: coeff = torch.tensor(coeff)
        return c + A*coeff

    def predicted_magnitude(self, A, M, d, coeff, pyt=True):
        if pyt:
            coeff = torch.tensor(coeff)
            logd = torch.log10(d*1e3/10.)
        else: logd = np.log10(d*1e3/10.)
        return M + A*coeff + 5.*logd

    def calc_logpx(self, model, x):
        # load data
        #x = toy_data.inf_train_gen(args.data, batch_size=batch_size)
        #x = torch.from_numpy(x).type(torch.float32).to(device)
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to z
        z, delta_logp = model(x, zero)

        # compute log q(z)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)

        logpx = logpz - delta_logp
        return logpx

    def get_transforms(self, model):

        def sample_fn(z, logpz=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z, reverse=True)

        def density_fn(x, logpx=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

        return sample_fn, density_fn


    def logjoint(self):

        distA = LogNormal(-0.5, 0.5)
        distd = HalfNormal(1.)
        distMC = NFdist(model = model, M=0., c=0.)

        A = pyro.sample("A", distA)#normal.Normal(-0.5, 0.5).log_prob(torch.log(A))
        d = pyro.sample("d", distd)
        Mc = pyro.sample('Mc', distMC)
 
        #c = Mc[:,1]*ss.scale_[1] + ss.mean_[1]
        #M = Mc[:,0]*ss.scale_[0] + ss.mean_[0]

        Mc_scaled = Mc*torch.tensor(self.ss.scale_) + torch.tensor(self.ss.mean_)
       #print(Mc_scaled, Mc)
        try:
            c = Mc_scaled[:,0:len(self.chat)]
            M = Mc_scaled[:,len(self.chat):]
        except IndexError:
            c = Mc_scaled[0:len(self.chat)]
            M = Mc_scaled[len(self.chat):]   
        #print(M, c, A, d)
        covc = torch.eye(len(self.sigmac))
        for i, s in enumerate(self.sigmac):covc[i,i]*=s**2 
        covm = torch.eye(len(self.sigmam))
        for i, s in enumerate(self.sigmam): covm[i,i]*=s**2
        #print(A*torch.tensor(dustco_c))
        #print(self.predicted_color(A, c, dustco_c))
        lnp_c = pyro.sample("chat", MultivariateNormal(self.predicted_color(A, c, self.dustco_c), covc), obs=torch.Tensor(self.chat))
        #lnp_c = pyro.sample("chat", Normal(self.predicted_color(A, c, dustco_c), self.sigmac), obs = torch.Tensor([self.chat]))
        lnp_m = pyro.sample("mhat", MultivariateNormal(self.predicted_magnitude(A, M, d, self.dustco_m), covm), obs=torch.Tensor(self.mhat))
        #lnp_m = pyro.sample("mhat", Normal(self.predicted_magnitude(A, M, d, dustco_m), self.sigmam), obs=torch.Tensor([self.mhat]))
        lnp_varpi = pyro.sample("varpihat", Normal(1./d, self.sigmavarpi), obs=torch.Tensor([self.varpihat]))
        lnp_Mc = distMC.log_prob(Mc)
        lnp_A = distA.log_prob(A)
        lnp_d = distd.log_prob(d)
        #print(lnp_c, lnp_m, lnp_varpi, lnp_Mc, lnp_A, lnp_d)
        #print(lnp_c[0], lnp_m[0], lnp_varpi[0], lnp_Mc[0][0], lnp_A, lnp_d, lnp_M[0], lnp_c[0])
        logp = torch.stack([lnp_c.sum(), lnp_m.sum(), lnp_varpi.sum(), lnp_Mc.sum(), lnp_A.sum(), lnp_d.sum()], dim=0).sum()
        return logp
    def lnprob_cmd(self, c, M):
    
        # Create a single array to feed to the model:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pdata = torch.zeros(1, len(c) + len(M), device=device)
        for i, p in enumerate(c):
            pdata[:,i] = (p - self.ss.mean_[i])/self.ss.scale_[i]
        for i, m in enumerate(M):
            pdata[:,-1] = (m - self.ss.mean_[-1])/self.ss.scale_[-1]

        # Query the model:
        model.eval()
        logprob = self.calc_logpx(model,pdata) #.cpu().detach()#.numpy()
        if torch.isnan(logprob): 
            #breakpoint()
            logprob=-1*torch.Tensor([float('Inf')]) #torch.Tensor([[-100]])
        #print(f'logp Mc {logprob}')
        return logprob
        #return torch.clamp(logprob, min=-100000)



class ObjectiveOpt(Objective):
    def __init__(self, theta, *args):
        super(ObjectiveOpt, self).__init__(*args)
        #print(torch.Tensor(theta[2]))
        self.A = nn.Parameter(torch.Tensor([theta[0]]), requires_grad=True)
        self.c = nn.Parameter(torch.Tensor(theta[1]), requires_grad=True)
        self.M = nn.Parameter(torch.Tensor(theta[2]), requires_grad=True)
        self.d = nn.Parameter(torch.Tensor([theta[3]]), requires_grad=True)
    


    def joint_hess(self, A, c0, c1, c2, c3, M0, d):
        #make each variable an input rather than combined in c
        c = [c0, c1, c2, c3]
        M = [M0]
        lnp_c = [normal.Normal(super(ObjectiveOpt, self).predicted_color(A, cmodel, coc), 
                               sigmacobs).log_prob(cobs) 
                               for cmodel, cobs, sigmacobs, coc 
                               in zip(c, self.chat, self.sigmac, self.dustco_c)]
        
        lnp_m = [normal.Normal(super(ObjectiveOpt, self).predicted_magnitude(A, Mmodel, d, com), 
                               sigmamobs).log_prob(mobs) 
                               for Mmodel, mobs, sigmamobs, com 
                               in zip(M, self.mhat, self.sigmam, self.dustco_m)]

        lnp_varpi = normal.Normal(1./d, self.sigmavarpi).log_prob(self.varpihat)
        lnp_Mc = self.lnprob_cmd(c, M)
        lnp_A = normal.Normal(0, 3.0).log_prob(A)
        lnp_d = normal.Normal(0, 1.).log_prob(d)
        #print('M:{:8.2f} G-J:{:8.2f} J-H:{:8.2f} H-K:{:8.2f} K-W1:{:8.2f} lnpMc:{:8.2f} A:{:8.2f} lnpA:{:8.2f} d:{:8.2f} lnpd:{:8.2f} lnpG-J:{:8.2f} lnpJ-H:{:8.2f} lnpH-K:{:8.2f} lnpK-W1:{:8.2f} lnpm:{:8.2f}, lnpvarpi:{:8.2f}'.format(M[0].item(), c[0].item(), c[1].item(), c[2].item(), c[3].item(), 
        #                                                                                                                            lnp_Mc.item(), A.item(), lnp_A.item(), d.item(), lnp_d.item(), 
        #                                                                                                                            lnp_c[0].item(), lnp_c[1].item(),lnp_c[2].item(),lnp_c[3].item(),lnp_m[0].item(), lnp_varpi.item()))

        #if torch.abs(A - torch.exp(torch.tensor(-3.))) < 0.01: 
        #    print('A near prior mode')
        logp = torch.stack([torch.stack(lnp_c).sum(), torch.stack(lnp_m).sum(), lnp_varpi.sum(), lnp_A.sum(), lnp_d.sum(), lnp_Mc.sum()], dim=0).sum()
        return torch.exp(logp)


    def logjoint(self, foo=None): #, A=None, G=None, G_J=None, J_H=None, H_K=None, K_W1=None, d=None):

        A = self.A
        M = self.M
        c = self.c
        d = self.d

        #if A is None: A = self.A
        #if G is None: G = self.M[0]
        #if G_J is None: G_J = self.c[0]
        #if J_H is None: J_H = self.c[1]
        #if H_K is None: H_K = self.c[2]
        #if K_W1 is None: K_W1 = self.c[3]
        #if d is None: d = self.d

        #beta_g, beta_bp, beta_rp = self.get_ext(self.M, self.c, torch.exp(self.lnA))
        #A = torch.exp(self.lnA)
        #M = self.M
        #c = self.c
        #d = torch.exp(self.lnd)
        #c = [G_J, J_H, H_K, K_W1]
        #M = [G]
        lnp_c = [normal.Normal(super(ObjectiveOpt, self).predicted_color(A, cmodel, coc), 
                               sigmacobs).log_prob(cobs) 
                               for cmodel, cobs, sigmacobs, coc 
                               in zip(c, self.chat, self.sigmac, self.dustco_c)]
        
        lnp_m = [normal.Normal(super(ObjectiveOpt, self).predicted_magnitude(A, Mmodel, d, com), 
                               sigmamobs).log_prob(mobs) 
                               for Mmodel, mobs, sigmamobs, com 
                               in zip(M, self.mhat, self.sigmam, self.dustco_m)]

        lnp_varpi = normal.Normal(1./d, self.sigmavarpi).log_prob(self.varpihat)
        #breakpoint()
        lnp_Mc = self.lnprob_cmd(c, M)
        lnp_A = uniform.Uniform(0, 10).log_prob(A)
        #print(f'A and lnA {A.item():8.3f} {lnp_A.item():8.3f}')
        #lnp_A = normal.Normal(0.0, 10.0).log_prob(A)
        lnp_d = normal.Normal(0, 1.).log_prob(d)
        #print('M:{:8.2f} G-J:{:8.2f} J-H:{:8.2f} H-K:{:8.2f} K-W1:{:8.2f} lnpMc:{:8.2f} A:{:8.2f} lnpA:{:8.2f} d:{:8.2f} lnpd:{:8.2f} lnpG-J:{:8.2f} lnpJ-H:{:8.2f} lnpH-K:{:8.2f} lnpK-W1:{:8.2f} lnpm:{:8.2f}, lnpvarpi:{:8.2f}'.format(M[0].item(), c[0].item(), c[1].item(), c[2].item(), c[3].item(), 
        #                                                                                                                            lnp_Mc.item(), A.item(), lnp_A.item(), d.item(), lnp_d.item(), 
        #                                                                                                                            lnp_c[0].item(), lnp_c[1].item(),lnp_c[2].item(),lnp_c[3].item(),lnp_m[0].item(), lnp_varpi.item()))

        #if torch.abs(A - torch.exp(torch.tensor(-3.))) < 0.01: 
            #print('A near prior mode')
            #print('logp c: {} c: {} c_true: {}'.format(lnp_c, c, '[1.06479639, 0.35893932, 0.04423247, 0.01361272]'))
            #print('logp m: {} M: {} M_true: {}'.format(lnp_m, M, '4.026'))
            #print('logp v: {} d: {} d_true: {}'.format(lnp_varpi, d, '2.558'))
            #print('logp Mc: {} lnp A: {} lnp d: {}'.format(lnp_Mc, lnp_A, lnp_d))
            #import pdb; pdb.set_trace()
        #lnp_beta_rp = 0. #norm.logpdf(beta_rp, 0.64, 0.01) #halfnorm(alpha, 0, 0.1)
        #lnp_beta_bp = 0. #norm.logpdf(beta_bp, 0.99, 0.01) #halfnorm(beta, 0, 0.1)
        #lnp_beta_g = 0. #norm.logpdf(beta_g, 0.73, 0.01)
        #print(A, M, c, d)
        #print(lnp_c, lnp_m, lnp_varpi, lnp_A, lnp_d, lnp_Mc)
        #import pdb; pdb.set_trace()
        logp = torch.stack([torch.stack(lnp_c).sum(), torch.stack(lnp_m).sum(), lnp_varpi.sum(), lnp_A.sum(), lnp_d.sum(), lnp_Mc.sum()], dim=0).sum()
        #print(lnp_c, lnp_m, lnp_varpi, lnp_A, lnp_d, lnp_beta_bp, lnp_beta_rp, lnp_beta_g, lnp_Mc)
        return logp
from scipy.stats import norm
from scipy.stats import uniform as uniformscipy
class NumericalObjective(Objective):
    def __init__(self, *args, model=None):
        super(NumericalObjective, self).__init__(*args)
        self.model = model    
    
    def logjoint(self, theta):
        A, c1, c2, c3, c4, M1, d = theta
        c = [c1, c2, c3, c4]
        M = [M1]
        lnp_c = np.sum([norm(super(NumericalObjective, self).predicted_color(A, cmodel, coc, pyt=False), 
                                   sigmacobs).logpdf(cobs) 
                                   for cmodel, cobs, sigmacobs, coc 
                                   in zip(c, self.chat, self.sigmac, self.dustco_c)])

        lnp_m = np.sum([norm(super(NumericalObjective, self).predicted_magnitude(A, Mmodel, d, com, pyt=False), 
                                   sigmamobs).logpdf(mobs) 
                                   for Mmodel, mobs, sigmamobs, com 
                                   in zip(M, self.mhat, self.sigmam, self.dustco_m)])

        lnp_varpi = norm(1./d, self.sigmavarpi).logpdf(self.varpihat)

        lnp_Mc = np.sum(self.lnprob_cmd(c, M).cpu().detach().numpy())
        lnp_A = uniformscipy(0, 10).logpdf(A)

        lnp_d = norm(0, 1.).logpdf(d)
        
        logp = np.sum([lnp_c, lnp_m, lnp_varpi, lnp_A, lnp_d, lnp_Mc])

        return logp

class NFdist(TorchDistribution):
    arg_constraints = {'M': constraints.real, 'c': constraints.real}
    support = constraints.real

    def __init__(self, model=None, M=0., c=1., validate_args=None):
        self.M, self.c = broadcast_all(M, c)
        self.model = model
        batch_shape = self.M.shape
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    def __call__(self, *args, **kwargs):
            """
            Samples a random value (just an alias for ``.sample(*args, **kwargs)``).

            For tensor distributions, the returned tensor should have the same ``.shape`` as the
            parameters.

            :return: A random value.
            :rtype: torch.Tensor
            """
            return self.sample(*args, **kwargs)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        #print(len(sample_shape))
        return self.model.sample(num_samples=1)


    #@dist.util.validate_sample
    def log_prob(self, value):
        lp = self.model(value)
        return lp
        #if torch.isnan(lp): lp=-torch.inf #torch.Tensor([[-8]])
        #return torch.clamp(lp, min=-10000)

def plot_marginal(ax, i, theta, sigma, c='black', lw=2, zorder=0):
    xmin, xmax = ax.get_xlim()
    xx = np.linspace(xmin, xmax, 100)
    if (i == 0) or (i == 3):
        xx += 0.0001
        ax.plot(xx, norm.pdf(np.log(xx), theta, sigma), c=c, lw=lw, zorder=zorder)
    else:
        ax.plot(xx, norm.pdf(xx, theta, sigma), c=c, lw=lw, zorder=zorder)

def plot_truth(theta_true, ax):
    numvars = len(theta_true)
    for i in range(0, numvars):
        true1 = theta_true[i]
        for j in range(0, numvars):
            true2 = theta_true[j]
            if i > j:
                continue
            elif i == j:
                ax[j,i].axvline(true1, c='red', zorder=np.inf, lw=2)
            else:
                ax[j,i].axvline(true1, c='red', zorder=np.inf, lw=2)
                ax[j,i].axhline(true2, c='red', zorder=np.inf, lw=2)

def plot_approx_posterior(theta_hat, sigma_hat, ax):
    numvars = len(theta_hat)
    for i in range(0, numvars):
        xlog = False 
        if (i == 0) or (i ==numvars-1): xlog=True
        mean1 = theta_hat[i]
        var1 = sigma_hat[i,i]

        for j in range(0, numvars):
            if i > j: continue
            else:
                ylog = False
                if (j == 0) or (j ==numvars-1): ylog=True
                mean2 = theta_hat[j]
                var2 = sigma_hat[j,j]
                var12 = sigma_hat[i,j]
                #import pdb; pdb.set_trace()
                if i == j:
                    loc = "right"
                    plot_marginal(ax[j,i], i, mean1, np.sqrt(var1), c='green', zorder=np.inf)
                else:
                    plotGaussian(mean1, mean2, var1, var12, var2, ax[j,i], lw=3, c='green', zorder=np.inf, xlog=xlog, ylog=ylog)

def plotGaussian(mu0, mu1, v00, v01, v11, ax, lw=1, c='k', zorder=100, xlog=False, ylog=False):
    V = np.zeros((2,2))
    mu = np.zeros(2)
    mu[0] = mu0
    mu[1] = mu1
    #sigma_hat = lnA, M, C, lnd
    V[0, 0] = v00
    V[0, 1] = v01
    V[1, 0] = v01
    V[1, 1] = v11
    GaussianContours(mu, V, ax, lw=lw, zorder=100, c=c, xlog=xlog, ylog=ylog)

def GaussianContours(mu, V, ax, amps=1., c='k', lw=1, label='prior', step=0.001, zorder=100, xlog=False, ylog=False):
    ts = np.arange(0, 2. * np.pi, step) #magic
    w, v = np.linalg.eigh(V)
    points = np.sqrt(w[0]) * (v[:, 0])[:,None] * (np.cos(ts))[None, :] + \
                np.sqrt(w[1]) * (v[:, 1])[:,None] * (np.sin(ts))[None, :] + \
                mu[:, None]
    if xlog: points[0,:] = np.exp(points[0,:])
    if ylog: points[1,:] = np.exp(points[1,:])
 
    ax.plot(points[0,:], points[1,:], c, lw=lw, alpha=amps/np.max(amps), rasterized=True, label=label, zorder=zorder)
    points = 2*np.sqrt(w[0]) * (v[:, 0])[:,None] * (np.cos(ts))[None, :] + \
                2*np.sqrt(w[1]) * (v[:, 1])[:,None] * (np.sin(ts))[None, :] + \
                mu[:, None]
    if xlog: points[0,:] = np.exp(points[0,:])
    if ylog: points[1,:] = np.exp(points[1,:])
    ax.plot(points[0,:], points[1,:], c, lw=lw, alpha=0.5, rasterized=True, label=label, zorder=zorder)

def add_cmd(fig):
    axcmd = fig.add_axes([0.6, 0.6, 0.3, 0.3]) 
    axcmd.pcolormesh(xx_cmd, yy_cmd, cmd_logp, cmap=plt.get_cmap('Blues'))
    axcmd.axhline(M_true[0], c='red')
    axcmd.axvline(c_true[0], c='red')
    axcmd.scatter(c[0], M[0], c='black')
    axcmd.scatter(theta_hat[1], theta_hat[-2], c='green')
    axcmd.scatter(np.median(samples['Mc'][:,0,1].detach().numpy()*ss.scale_[1] + ss.mean_[1]), 
                  np.median(samples['Mc'][:,0,-2].detach().numpy()*ss.scale_[-2] + ss.mean_[-2]), c='orange')
    axcmd.invert_yaxis()                                                                                        
                                                                                                           

def optimize_model(theta_0, chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss, i):
    objective = ObjectiveOpt(theta_0, chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss)

    obj = PyTorchObjective(objective, f'gradients_{i}.txt')

    maxiter=1000
    method="L-BFGS-B" # or SLSQP"Newton-CG"#'BFGS' #, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
    res = minimize(obj.fun, obj.x0, method=method, callback=None, jac=obj.jac,
                    options = {'xtol': 1e-7, 'disp': False, 'maxiter':maxiter})#gobj) ,  hess=obj.hes, 
    theta_hat = res.x
    input = (torch.tensor(theta_hat[0]), 
            torch.tensor(theta_hat[1]),
            torch.tensor(theta_hat[2]),
            torch.tensor(theta_hat[3]),
            torch.tensor(theta_hat[4]),
            torch.tensor(theta_hat[5]), 
            torch.tensor(theta_hat[6]))
    #hes = torch.autograd.functional.hessian(objective.joint_hess, input)
    #hes1 = np.zeros((7,7))
    #for i in range(7): hes1[i,:] = hes[i]
    #hes = torch.autograd.functional.hessian(objective.logjoint, torch.zeros(1)) 
    hes2 = hessianrepo(objective.logjoint(), objective.parameters()) #, create_graph=True)
    """
x = torch.ones(4, requires_grad=True) * 2
print(jacobian2(f(x), x, 2))
a = hessian(f(x), x, 2)
print(a)
"""
    #from IPython import embed; embed()

    try: 
        #sigma_hat = np.linalg.inv(-1.*hes1)
        sigma_hat2 = np.linalg.inv(-1.*hes2.detach().numpy())
    except np.linalg.LinAlgError:
        sigma_hat2 = np.ones(hes.shape)*np.inf
    return res, sigma_hat2

import numdifftools as nd
def numerical_hessian(theta_hat, chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss):
    #print(theta_hat)
    numobjective = NumericalObjective(chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss)
    logjoint = numobjective.logjoint(theta_hat)
    #print(logjoint)
    hessian = nd.Hessian(numobjective.logjoint)
    return hessian(theta_hat)


def sample_model(chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, theta_0_mcmc, nsamples=100, nwalkers=1):
    objective = Objective(chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m)
    #print(objective.logjoint())
    objective.logjoint()
    #nuts_kernel = NUTS(objective.logjoint, jit_compile=True, ignore_jit_warnings=True)
    #mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=100, num_chains=2, mp_context='spawn')

    try:
        with open('savemcmc_{}.pkl'.format(ind), 'rb') as f:
            mcmc = pickle.load(f)
    except IOError:
        nuts_kernel = NUTS(objective.logjoint, jit_compile=True, ignore_jit_warnings=False)
        mcmc = MCMC(nuts_kernel, num_samples=nsamples, warmup_steps=100, num_chains=nwalkers, initial_params=theta_0_mcmc, mp_context='spawn') 
        mcmc.run()

        with open('savemcmc_{}.pkl'.format(ind), 'wb') as f:
            mcmc.sampler = None
            mcmc.kernel.potential_fn = None
            pickle.dump(mcmc, f)
    return mcmc, objective

def plot1d():
    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    ax[0].axvline(teststars['AV_0'][ind], lw=2, label='{:.2f}'.format(np.exp(theta_hat[0])))
    lnA = np.linspace(-3, 0, 100)
    ax[0].plot(np.exp(lnA), norm.pdf(lnA, theta_hat[0], np.sqrt(sigma_hat[0,0])))
    ax[0].axvline(np.exp(theta_hat[0]), lw=2, c='green')
    ax[0].set_xlabel('A')


    ax[1].axvline(teststars['distance_0'][ind]/1e3 , lw=2, label='{:.2f}'.format(np.exp(theta_hat[-1])))
    lnd = np.linspace(-3, 2, 100)
    ax[1].plot(np.exp(lnd), norm.pdf(lnd, theta_hat[3], np.sqrt(sigma_hat[-1,-1])))
    ax[1].axvline(np.exp(theta_hat[-1]), lw=2, c='green')
    ax[1].set_xlabel('d')

    m_hat = theta_hat[1+len(c_true):-1]
    m_sig = sigma_hat[1+len(c_true):-1,1+len(c_true):-1]
    for i, mag in enumerate(m_hat):
        ax[2].axvline(M_true[i], lw=2, label='{:.2f}'.format(m_hat[i]))
        M = np.linspace(-5, 10, 100)
        #M = np.linspace(theta_hat[1] - np.sqrt(sigma_hat[1,1])*3., 
        #                  theta_hat[1] + np.sqrt(sigma_hat[1,1])*3, 100)
        ax[2].axvline(m_hat[i], lw=2, c='green')
        ax[2].plot(M, norm.pdf(M, m_hat[i], np.sqrt(m_sig[i,i])))
        ax[2].set_xlabel('M')
    
    c_hat = theta_hat[1:1+len(c_true)]
    c_sig = sigma_hat[1:1+len(c_true)]
    for i, col in enumerate(c_hat):
        ax[3].axvline(c_true[i], lw=2, label='{:.2f}'.format(c_hat[i]))
        ax[3].axvline(c_hat[i], lw=2, c='green')
        c = np.linspace(-0.5, 2, 100)
        #c = np.linspace(theta_hat[2] - np.sqrt(sigma_hat[2,2])*3., 
        #                  theta_hat[2] + np.sqrt(sigma_hat[2,2])*3, 100)
        ax[3].plot(c, norm.pdf(c, c_hat[i], np.sqrt(c_sig[i, i])))
        ax[3].set_xlabel('c')
        #A_hat, A_sigma = meansig_lognorm(theta_hat, sigma_hat, 0)
        #d_hat, d_sigma = meansig_lognorm(theta_hat, sigma_hat, 3)
    for a in ax: a.legend()
    phot = samples['Mc']*ss.scale_ + ss.mean_
    ax[0].hist(samples['A'], bins=20, histtype='step', density=True)
    ax[1].hist(samples['d'], bins=20, histtype='step', density=True)
    ax[2].hist(phot[:,0, -1], bins=20, histtype='step', density=True)
    for i in range(len(c_true)):
        ax[3].hist(phot[:,0,i], bins=20, histtype='step', density=True)
    
    

    plt.savefig('mcmc_{}.pdf'.format(ind))
    #plt.show()
    plt.close(fig)

def plot2d(samples, theta_true, theta_hat, sigma_hat):
    data = np.zeros((len(samples['A']), len(theta_true)))
    data[:,0] = samples['A'].detach().numpy()
    data[:,-1] = samples['d'].detach().numpy()
    for i in range(5):
        data[:,1+i] = samples['Mc'][:,0,i].detach().numpy()*ss.scale_[i] + ss.mean_[i]
    names = ['A', 'G-H', 'G-J', 'J-K', 'K-W1', 'G', 'd']
    ranges = [[np.min(data[:,i]), np.max(data[:,i])] for i in np.arange(data.shape[-1])]
    figure = corner.corner(data, labels=names, truths=theta_true, range=ranges)
    axes = np.array(figure.axes).reshape(len(theta_true), len(theta_true))

    axcmd = figure.add_axes([0.6, 0.6, 0.3, 0.3]) 
    axcmd.pcolormesh(xx_cmd, yy_cmd, cmd_logp, cmap=plt.get_cmap('Blues'))
    axcmd.axhline(M_true[0], c='red')
    axcmd.axvline(c_true[0], c='red')
    axcmd.scatter(c[0], M[0], c='black')
    axcmd.scatter(theta_hat[1], theta_hat[-2], c='green')
    axcmd.scatter(np.median(data[:, 1]), 
                  np.median(data[:,-2]), c='orange')
    axcmd.invert_yaxis()

    #add_cmd(figure)
    plot_truth(theta_true, axes)
    plot_approx_posterior(theta_hat, sigma_hat*1e4, axes)
    figure.savefig('mcmc2d_{}.pdf'.format(ind))

"""    
def plot2d(objective, mcmc, theta_true, theta_hat, sigma_hat):
    prior = Predictive(objective.logjoint, {}, num_samples=200).get_samples()
    pyro_data = az.from_pyro( mcmc, prior=prior)#,#posterior_predictive=posterior_predictive)

    #import pdb; pdb.set_trace()
    axes2, fig2 = az.plot_pair(pyro_data, kind=['scatter', 'kde'],
                diagonal=True, point_estimate='median',
                coords = {"Mc_dim_1": [0, 1]})
    add_cmd(fig2)
    import pdb; pdb.set_trace()
    print(axes2.shape())
    import pdb; pdb.set_trace()
    plot_truth(theta_true, axes2) 


    plot_approx_posterior(theta_hat, sigma_hat, axes2)
    fig2.savefig('mcmc2d_{}.pdf'.format(ind))
    plt.close(fig2)
"""



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)
scale=10; fraction=0.1
model = gen_model(scale=scale, fraction=fraction)
model.eval()
model.requires_grad_(False)
#hack to get sample working, log_probs needs to be called first
#foo = torch.zeros(1, 5, device=device)

#model.log_probs(foo)

#from pyro.nn.module import to_pyro_module_
#to_pyro_module_(model)
model.requires_grad_(False)
#


if __name__ == '__main__':
    freeze_support()

    from scipy.stats import norm
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float64)
    numdatasamples = 1000000
    ss = pkl.load(open(f'transform_nsamples{numdatasamples}.pkl','rb'))
    samples = pkl.load(open(f'fullpop_nsamples{numdatasamples}.pkl', 'rb')) #model.sample(num_samples=10000).detach().numpy()

    #cmd_logp, xx_cmd, yy_cmd = make_cmdlogp_plot.make()

    import time
    t = time.time()
    teststars = pd.read_csv('teststars_335.csv')
    teststars_nodust = pd.read_csv('teststars_335_nodust.csv')
    bands = ['G_mag', 'J_mag', 'H_mag', 'K_mag', 'W1_mag']

    dust_coeff = {}

    for b in bands:
        coeff_band = (teststars[b] - teststars_nodust[b])/teststars['AV_0']
        dust_coeff[b] = np.median(coeff_band)


    for pop in [teststars, teststars_nodust]:
        for band1, band2 in zip(bands[:-1], bands[1:]):
            b1 = band1.split('_')[0]
            b2 = band2.split('_')[0]
            pop['{0}-{1}'.format(b1, b2)] = pop[band1] - pop[band2]
            dust_coeff['{0}-{1}'.format(b1, b2)] = dust_coeff[band1] - dust_coeff[band2]

    use_bands = ['G_mag', 'J_mag', 'H_mag', 'K_mag', 'W1_mag']
    color_bands = ['G_mag', 'J_mag', 'H_mag', 'K_mag', 'W1_mag']
    color_keys = ['G-J', 'J-H', 'H-K', 'K-W1']
    absmag_keys = ['G_mag']

    use_cols = ['G-J', 'J-H', 'H-K', 'K-W1', 'G_mag']
    #use_cols = ['G_mag', 'bp_rp']#, 'BP', 'RP']
    cond_cols = [] # ['age', 'feh'] #, 'alpha', 'fB', 'gamma']

    dustco_c = [dust_coeff[b] for b in color_keys]    
    dustco_m = [dust_coeff[b] for b in absmag_keys]


    for ind in range(10): # range(len(teststars['AV'])):
        ind += 2
        t = time.time()
        distance = teststars['distance_0'][ind]/1e3 
        A = teststars['AV_0'][ind]

        M = [teststars[m][ind] - 5.*np.log10(distance*1e3/10) for m in absmag_keys]
        c = [teststars[c][ind] for c in color_keys]

        M_true = [teststars_nodust[m][ind] - 5.*np.log10(distance*1e3/10) for m in absmag_keys]
        c_true = [teststars_nodust[m][ind] for m in color_keys]

        sigmac = [0.01] * len(c)
        sigmam = [0.01] * len(M)
        sigmavarpi = 0.01
        
        chat = [color + rs.normal()*sigmacobs for color, sigmacobs in zip(c, sigmac)] #(scale=sigmac)
        varpihat = 1/distance + rs.normal()*sigmavarpi #(scale=sigmavarpi)
        mhat = [absm + 5*np.log10(distance*1e3/10.) + rs.normal()*sigmamobs for absm, sigmamobs in zip(M, sigmam)] #(scale=sigmam)
        print('True A is: ', A)
        #theta_0 for optimization [lnA, c, M, lnd]
        theta_0 = [torch.log(torch.from_numpy(np.array(A))),
                             torch.from_numpy(np.array(c_true)),
                             torch.from_numpy(np.array(M_true)),
                   torch.log(torch.from_numpy(np.array(distance)))]

        res, sigma_hat = optimize_model(theta_0, chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss, ind)
        theta_hat = res.x
        #theta_0 for mcmc []
        nwalkers=2
        nsamples=500
        A_0 = torch.zeros(nwalkers, 1)
        A_0 = torch.from_numpy(np.exp(theta_hat[0]) + 1e-4 * np.random.randn(nwalkers))
        Mc_0 = torch.zeros(nwalkers, 1, 5)
        Mc_0[:,0,:] = torch.from_numpy((theta_hat[1:-1]-ss.mean_)/ss.scale_ + 1e-4 * np.random.randn(nwalkers, 5))
        d_0 = torch.zeros(nwalkers, 1)
        d_0 = torch.from_numpy(np.exp(theta_hat[-1]) + 1e-4 * np.random.randn(nwalkers))
        theta_0_mcmc = {'A':torch.from_numpy(np.array(A_0)), 
                        'Mc':torch.from_numpy(np.array(Mc_0)),
                        'd': torch.from_numpy(np.array(d_0))}

        mcmc, objective = sample_model(chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, theta_0_mcmc, nsamples=nsamples, nwalkers=nwalkers)

        samples = mcmc.get_samples()

        #import pdb;pdb.set_trace()        

        theta_true = [A] + list(c_true) + list(M_true) + [distance]
        theta_inf = [np.exp(theta_hat[0])] + list(theta_hat[1:len(color_keys) + 1]) + [np.exp(theta_hat[-1])]
        #print(theta_true, theta_inf)
        plot1d()
        plot2d(samples, theta_true, theta_hat, sigma_hat)

        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()


    #if __name__ == '__main__': main()