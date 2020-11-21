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

from hessian import hessian 
import pickle
#import pyro
#from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform, LogNormal, HalfNormal, MultivariateNormal
#from pyro.distributions.util import scalar_like
#from pyro.infer import MCMC, NUTS, Predictive
#from pyro.infer.mcmc.util import initialize_model, summary
#from pyro.util import ignore_experimental_warning

from torch.multiprocessing import Pool, set_start_method, freeze_support

from torch.distributions import constraints, normal, half_normal
from torch.distributions.utils import broadcast_all
#from pyro.distributions import TorchDistribution
#
#import arviz as az
#import make_cmdlogp_plot

import corner

#from pyro_torch_model_5D_linearA import gen_model, ObjectiveOpt, plot_marginal, plot_truth, plot_approx_posterior, add_cmd, optimize_model
from pyro_numerical_derivative import gen_model, ObjectiveOpt, plot_marginal, plot_truth, plot_approx_posterior, add_cmd, optimize_model, numerical_hessian 


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

def plot2d(theta_true, theta_hat, sigma_hat):
    fig, axes = plt.subplots(7,7, figsize=(10,10))
    names = ['A', 'G-H', 'G-J', 'J-K', 'K-W1', 'G', 'd']
    plot_truth(theta_true, axes)
    plot_approx_posterior(theta_hat, sigma_hat, axes)
    axcmd = fig.add_axes([0.6, 0.6, 0.3, 0.3]) 
    axcmd.pcolormesh(xx_cmd, yy_cmd, cmd_logp, cmap=plt.get_cmap('Blues'))
    axcmd.axhline(M_true[0], c='red')
    axcmd.axvline(c_true[0], c='red')
    axcmd.scatter(c[0], M[0], c='black')
    axcmd.scatter(theta_hat[1], theta_hat[-2], c='green')
    axcmd.invert_yaxis()
    for i in range(7):
        for j in range(7):
            axes[i,j].set_xlim(theta_true[j]*0.5, theta_true[j]*1.5)
            axes[i,j].set_ylim(theta_true[i]*0.5, theta_true[i]*1.5)
            if j == 6: axes[i,j].set_xlabel(names[i])
            if i == 0: axes[i,j].set_ylabel(names[i])
            if j > i: axes[i,j].remove()
    plt.tight_layout()
    fig.savefig('mcmc2d_optimized_{}.pdf'.format(ind))




device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)
scale=10
fraction=0.1
model = gen_model(scale=scale, fraction=fraction)
model.eval()
model.requires_grad_(False)
#hack to get sample working, log_probs needs to be called first
#0foo = torch.zeros(1, 5, device=device)

#model.log_probs(foo)

#from pyro.nn.module import to_pyro_module_
#to_pyro_module_(model)
model.requires_grad_(False)
numdatasamples = 1000000
ss = pickle.load(open(f'transform_nsamples{numdatasamples}.pkl','rb'))

#ss = pickle.load(open(f'transform_cycle_gauss_scale{scale}_frac{fraction}.pkl', 'rb')) #'transform.pkl','rb'))
#cmd_logp, xx_cmd, yy_cmd = make_cmdlogp_plot.make()
#
if __name__ == '__main__':
    from astropy.table import Table
    from astropy.io import ascii
    np.random.seed(222)
    torch.random
    freeze_support()

    from scipy.stats import norm
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float64)

    #samples = model.sample(num_samples=10000).detach().numpy()
    #corner.corner(samples)

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
    print(dustco_c, dustco_m)

    fullpop = pd.read_pickle('fullpop.pkl')
    c0 = np.median(fullpop[color_keys])
    M0 = np.median(fullpop[absmag_keys])


    nstars = 335
    #nstars = 50
    arr = {'A_true': np.zeros(nstars),
           'd_true': np.zeros(nstars), 
           'M_true': np.zeros(nstars),
           'c_true': np.zeros((nstars, len(color_keys))),
           'varpi_obs': np.zeros(nstars),
           'm_obs':np.zeros(nstars),
           'c_obs':np.zeros((nstars, len(color_keys))),
           'varpi_sig':np.zeros(nstars),
           'm_sig':np.zeros(nstars),
           'c_sig': np.zeros((nstars, len(color_keys))),
           'A': np.zeros(nstars),
           'd': np.zeros(nstars),
           'M': np.zeros(nstars), 
           'c': np.zeros((nstars, len(color_keys))),
           'cov':np.zeros((nstars, len(color_keys) + len(absmag_keys) + 2, len(color_keys) + len(absmag_keys) + 2)),
           'hes':np.zeros((nstars, len(color_keys) + len(absmag_keys) + 2, len(color_keys) + len(absmag_keys) + 2)),
           'res':[]}

    for ind in range(nstars): #[3,  5,   7,   8,  11,  12,  13]: #4 range(nstars): # range(len(teststars['AV'])):
        #ind = 3
        t = time.time()
        distance = teststars['distance_0'][ind]/1e3 
        A = teststars['AV_0'][ind]

        M_true = [teststars_nodust[m][ind] - 5.*np.log10(distance*1e3/10) for m in absmag_keys]
        c_true = [teststars_nodust[m][ind] for m in color_keys]

        M = [teststars[m][ind] - 5.*np.log10(distance*1e3/10) for m in absmag_keys]
        c = [teststars[c][ind] for c in color_keys]

        #M = [teststars_nodust[m][ind] - 5.*np.log10(distance*1e3/10) + A*dust_coeff[m] for m in absmag_keys]
        #c = [teststars_nodust[m][ind] + A*dust_coeff[m] for m in color_keys]

        sigmac = [0.01, 0.01, 0.001, 0.001] 
        sigmam = [0.01] * len(M)
        sigmavarpi = 0.05
        
        chat = [color + rs.normal()*sigmacobs for color, sigmacobs in zip(c, sigmac)] #(scale=sigmac)
        varpihat = 1/distance + rs.normal()*sigmavarpi #(scale=sigmavarpi)
        mhat = [absm + 5*np.log10(distance*1e3/10.) + rs.normal()*sigmamobs for absm, sigmamobs in zip(M, sigmam)] #(scale=sigmam)

        arr['A_true'][ind] = A 
        arr['d_true'][ind] = distance
        arr['M_true'][ind] = M_true[0]
        arr['c_true'][ind,:] = c_true
        arr['varpi_obs'][ind] = varpihat
        arr['m_obs'][ind] = mhat[0]
        arr['c_obs'][ind] = chat 
        arr['varpi_sig'][ind]=sigmavarpi
        arr['m_sig'][ind] = sigmam[0]
        arr['c_sig'][ind,:] = sigmac

        #theta_0 for optimization [lnA, c, M, lnd]
        """
        theta_0 = [torch.log(torch.from_numpy(np.array(0.1) + 0.1 * np.random.randn(1))),
                             torch.from_numpy(np.array(chat)+ 0.1 * np.random.randn(len(color_keys))),
                             torch.from_numpy(np.array(mhat - 5*np.log10(1e2/varpi))+ 0.1 * np.random.randn(len(absmag_keys))),
                   torch.log(torch.from_numpy(np.array(1/varpihat)+ 0.1 * np.random.randn(1)))]
        """
        A0 = 0.1 #A
        theta_0 = [          torch.from_numpy(np.array(A0)), #torch.log()
                             torch.from_numpy(np.array(chat) - A0*np.array(dustco_c)),
                             torch.from_numpy(np.array(mhat - 5*np.log10(1e2/varpihat) - A0*np.array(dustco_m))),
                             torch.from_numpy(np.array(1/varpihat))]

        """
        theta_0 = [          torch.from_numpy(np.array(A) + 1e-1 * np.random.randn(1)), #torch.log()
                             torch.from_numpy(np.array(c_true)+ 1e-2 * np.random.randn(len(color_keys))),
                             torch.from_numpy(np.array(M_true)+ 1e-2 * np.random.randn(len(absmag_keys))),
                             torch.from_numpy(np.array(distance)+ 1e-1 * np.random.randn(1))]
        """
        print('###############################################################')
        print('###############################################################')
        print('###############################################################')
        print(f'theta_0 is: {theta_0}')
        print(f'A true is: {A}')
        
        #hessian = numerical_hessian(np.hstack([t.cpu().detach().numpy().astype(np.float64) for t in theta_hat]), chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss)
        res, sigma_hat = optimize_model(theta_0, chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss, ind)
        theta_hat = res.x
        hessian = numerical_hessian(np.hstack(theta_hat), chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss)
        sigma_hat_num = np.linalg.inv(-1.*hessian)
        print(f'A true is {A:.4f}, A infered is {theta_hat[0]:.4f}, difference is {A - theta_hat[0]:.4f}')
        print(f'sigma A pytorch is {np.sqrt(sigma_hat[0,0]):.3f} and sigma A numerical is {np.sqrt(sigma_hat_num[0,0]):.3f}')
        #hessian = (theta_hat, chat, mhat, varpihat, sigmac, sigmam, sigmavarpi, dustco_c, dustco_m, ss)
        arr['res'].append(res)
        #import pdb; pdb.set_trace()
        arr['A'][ind] = theta_hat[0] #np.exp(theta_hat[0])
        arr['d'][ind] = theta_hat[-1] #np.exp(theta_hat[-1])
        arr['c'][ind,:] = theta_hat[1:1+len(color_keys)]
        arr['M'][ind] = theta_hat[-2]
        arr['cov'][ind, :,:] = sigma_hat_num
        arr['hes'][ind, :, :] = hessian


        theta_true = [A] + list(c_true) + list(M_true) + [distance]
        #theta_hat[0] = np.exp(theta_hat[0])
        #theta_hat[-1] = np.exp(theta_hat[-1])
        #plot1d()
        #print([np.sqrt(sigma_hat[i,i]) for i in range(7)])
        #plot2d(theta_true, theta_hat, sigma_hat)

    t = Table(arr)
    np.save('optvalues_{}_linearA_nograd.npy'.format(nstars), t)
    #import pdb; pdb.set_trace()
    #ascii.write(t, 'optvalues.dat', format='ascii')
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()


    #if __name__ == '__main__': main()