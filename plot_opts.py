import numpy as np
import matplotlib.pyplot as plt 

params = {'legend.fontsize': 'x-small',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-small',
         'axes.titlesize':'x-small',
         'xtick.labelsize':'x-small',
         'ytick.labelsize':'x-small'}
plt.rcParams.update(params)


def meansig_lognorm(theta_hat, sigma_hat, index):
    mean = np.exp(theta_hat[index] + sigma_hat[index,index]/2.)
    var = (np.exp(sigma_hat[index, index]) - 1)*np.exp(2*theta_hat[index] + sigma_hat[index, index])
    return mean, np.sqrt(var)
def sig_lognorm(mu, var):
    return np.sqrt((np.exp(mu) - 1)*np.exp(2*mu + var))

def mean_lognorm(mu, var):
    return np.exp(mu + var/2.)

def plot_A_d(data):
    #A_sig = [sig_lognorm(data['A'][i], data['cov'][i,0,0]) for i in range(len(data))]
    mse = np.sum((data['A'] - data['A_true'])**2)/len(data)
    mse_d = np.sum((data['d'] - data['d_true'])**2)/len(data)
    mse_do = np.sum((1./data['varpi_obs'] - data['d_true'])**2)/len(data)
    A_sig = [np.sqrt(data['cov'][j,0,0]) for j in range(len(data))]
    #plt.hist(A_sig, bins=np.linspace(0, 5, 100), log=True)
    #plt.show()
    #plt.scatter(data['A_true'], data['A'])
    apx = np.linspace(0, 5, 100)
    fig, axes = plt.subplots(3, figsize=(10,10))
    ax = axes[0]
    ax.errorbar(data['A'], data['A'] - data['A_true'], yerr = A_sig, fmt='o', alpha=0.2)
    #ax.plot([0, 5], [0, 5], c='black', lw=2)
    ax.axhline(0)
    ax.set_xlim(0, 4)
    #ax.set_ylim(-1, 6)
    ax.set_xlabel('A_true')
    ax.set_ylabel('A - A_true')
    fig.suptitle(f'mse A {mse:4.4f} mse d {mse_d:4.4f} mse d_obs {mse_do:4.4f}', fontsize=12)
    #ax.savefig('A_Atrue.pdf', rasterized=True)

    ax = axes[1]
    ax.errorbar(data['d_true'], data['d'] - data['d_true'], yerr = np.sqrt(data['cov'][:,-1,-1]), fmt='o', alpha=0.2)
    #ax.plot([0, 3], [0, 3], c='black', lw=2)
    ax.axhline(0)
    ax.set_xlim(0, 3)
    #ax.set_ylim(-2, 5)
    ax.set_xlabel('d_true')
    ax.set_ylabel('d')

    ax = axes[2]
    badsig = np.logical_or(np.isnan(A_sig), np.isinf(A_sig))
    ax.scatter(data['A_true'][badsig], (data['A'] - data['A_true'])[badsig], c='black', alpha=0.2)
    ax.scatter(data['A_true'], (data['A'] - data['A_true'])/A_sig, alpha=0.2)
    ax.axhline(1, alpha=0.5); ax.axhline(-1, alpha=0.5) 
    ax.axhline(2, alpha=0.25); ax.axhline(-2, alpha=0.25) 
    ax.axhline(3, alpha=0.1); ax.axhline(-3, alpha=0.1)
    ax.set_ylabel('z score')
    ax.set_xlabel('A_true')
    plt.tight_layout()
    fig.savefig('A_Atrue_log.pdf', rasterized=True)

def plot_corner(data, columns):
    index = np.abs(data['A_true'] - data['A']) < 0.25
    fig, ax = plt.subplots(3, 3)
    ax = ax.ravel()
    names = ['G-J', 'J-H', 'H-K', 'K-W1']    
    """
    for co, c, a, n in zip(data['c_true'].T, data['c'].T, ax, names):
        a.plot(co[index], c[index], 'ko', markersize=10, label=n)  
        a.plot(co[~index], c[~index], 'go', markersize=10)
        a.legend() 
        a.set_xlabel('observed')
        a.set_ylabel('inferred')
    for mo, m, a, n in zip([data['m_obs'], data['m_obs']], 
                        [data['M'] + 5*np.log10(data['d']*1e3/10),  
                            data['M'] + 5*np.log10(data['d_true']*1e3/10)], 
                        [ax[5], ax[6]],
                        ['M infd', 'M trued']):
        a.plot(mo[index], m[index], 'ko', markersize=10, label=n)  
        a.plot(mo[~index], m[~index], 'go', markersize=10)
        a.legend() 
    plt.show()
    """

    fig, ax = plt.subplots(len(columns)-2, len(columns)-2, figsize=(20,20))
    for i, k in enumerate(columns[:-2]):
        if k == 'cov': continue
        if k == 'res': continue
        for j, kk in enumerate(columns[:-2]):
            if kk == 'cov': continue
            if kk == 'res': continue
            if k == kk: 
                try:
                    ax[i,j].hist(data[k][index], color='k', histtype='step')
                    ax[i,j].hist(data[k][~index], color='green', histtype='step')
                except ValueError:
                    ax[i,j].hist(data[k][index], color=['k']*4, histtype='step')
                    ax[i,j].hist(data[k][~index], color=['green']*4, histtype='step')

            else:
                ax[i,j].plot(data[kk][~index], data[k][~index], 'go', markersize=1)
                ax[i,j].plot(data[kk][index], data[k][index], 'ko', markersize=1)
                

            if i == len(columns)-3: 
                ax[i,j].set_xlabel(kk)
                ax[i,j].set_yticklabels([])
            elif j == 0: 
                ax[i,j].set_ylabel(k)
                ax[i,j].set_xticklabels([])
            else: 
                ax[i,j].set_yticklabels([])
                ax[i,j].set_xticklabels([])
            if j > i: ax[i,j].remove()
    plt.tight_layout()
    fig.savefig(f'corner_opt_{nstars}.pdf', rasterized=True)  

def plot_cmd(data, samples):
    size = 10
    sigA = np.array([np.sqrt(c[0,0]) for c in data['cov']])
    badsig = np.logical_or(np.isnan(sigA), np.isinf(sigA))
    sigA[badsig] = 10000.
    index = np.abs(data['A_true'] - data['A']) < 3.*sigA
    index1  = data['A_true'] - data['A'] >= 3.*sigA
    index2 = data['A_true'] - data['A'] <= -3.*sigA
    #index3 = np.abs(data['A'] - np.log(-3)) < 0.01
    plt.clf()
    fig, ax = plt.subplots(2, 3, figsize=(10, 7))
    ax = ax.ravel()

    ax[0].scatter(data['A_true'][index], data['A'][index], s=size)
    ax[0].scatter(data['A_true'][index1], data['A'][index1], c='black', label='under', s=size)
    ax[0].scatter(data['A_true'][index2], data['A'][index2], edgecolors='black', facecolors='none', label='over', s=size)
    #ax[0].scatter(data['A_true'][badsig], data['A'][badsig], c='green', s=size)
    ax[0].set_xlabel('A_true')
    ax[0].plot([-1, 6], [-1, 6], c='black', lw=2)
    ax[0].set_ylabel('A')
    ax[0].legend()

    ax[1].scatter(data['d_true'][index], data['d'][index], s=size)
    ax[1].scatter(data['d_true'][index1], data['d'][index1], c='black', label='under', s=size)
    ax[1].scatter(data['d_true'][index2], data['d'][index2], edgecolors='black', facecolors='none', label='over', s=size)
    ax[1].set_xlabel('d_true')
    ax[1].set_ylabel('d')
    ax[1].plot([0, 3], [0, 3], c='black', lw=2)
    ax[1].legend()


    ax[2].scatter(data['c_true'][:, 0], data['M_true'], c='red', label='truth')
    ax[2].scatter(data['c'][:, 0][index], data['M'][index], s=size, label='infer')
    ax[2].scatter(data['c'][:, 0][index1], data['M'][index1], c='black', s=size, label='under')
    ax[2].scatter(data['c'][:, 0][index2], data['M'][index2], edgecolors='black',facecolors='none', s=size, label='over')

    ax[2].legend()
    ax[2].set_xlabel('G-J')
    ax[2].set_ylabel('G')
    ax[2].invert_yaxis()
    xlim = ax[2].get_xlim(); ylim=ax[2].get_ylim()
    ax[2].plot(samples[:,0], samples[:,-1], 'o', color='grey', alpha=0.1, markersize=2, zorder=-1)
    ax[2].set_xlim(xlim); ax[2].set_ylim(ylim)
    labels = ['G-J', 'J-H', 'H-K', 'K-W1']
    for (a, c1, c2, l1, l2, ct1, ct2) in zip(ax[3:], 
                                            data['c'].T[:-1], 
                                            data['c'].T[1:],
                                            labels[:-1], 
                                            labels[1:], 
                                            data['c_true'].T[:-1], 
                                            data['c_true'].T[1:]):
        a.scatter(ct1, ct2, c='red', label='truth')
        a.scatter(c1[index], c2[index], s=size, label='infer')
        a.scatter(c1[index1], c2[index1], c='black', s=size, label='under')
        a.scatter(c1[index2], c2[index2], edgecolors='black', s=size, facecolors='none', label='over')
        xlim = a.get_xlim(); ylim=a.get_ylim()
        #a.scatter(s1, s2, c='grey', alpha=0.1, zorder=-1, edgecolor='none')
        a.set_xlim(xlim); a.set_ylim(ylim)
        a.set_xlabel(l1)
        a.set_ylabel(l2)
        a.legend()

    ax[3].plot(samples[:,0], samples[:,1], 'o', color='grey', alpha=0.1, markersize=2, zorder=-1)
    #ax[3].set_xlim(-0.01, 3.5)
    #ax[3].set_ylim(-0.02, 1)

    ax[4].plot(samples[:,1], samples[:,2], 'o', color='grey', alpha=0.1, markersize=2, zorder=-1)
    #ax[4].set_ylim(-0.11, .35)
    #ax[4].set_xlim(-0.02, 1)

    ax[5].plot(samples[:,2], samples[:,3], 'o', color='grey', alpha=0.1, markersize=2, zorder=-1)
    #ax[5].set_xlim(-0.11, .35)
    #ax[5].set_ylim(-0.11, .3)
    #ax[3].scatter(samples[:,0], samples[:,1], c='grey', alpha=0.1, zorder=-1, edgecolor='none')
    #ax[4].scatter(samples[:,1], samples[:,2], c='grey', alpha=0.1, zorder=-1, edgecolor='none')
    #ax[5].scatter(samples[:,2], samples[:,3], c='grey', alpha=0.1, zorder=-1, edgecolor='none')
    plt.tight_layout()
    fig.savefig(f'cmd_a_opt_{nstars}.pdf', rasterized=True)

def get_transforms(model):

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

if __name__ =='__main__':
    nstars = 30
    data = np.load(f'optvalues_{nstars}_linearA.npy', allow_pickle=True)

    columns = data.dtype.names

    plot_corner(data, columns)
    plot_A_d(data)
    import pandas as pd
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import torch
    import pickle
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float64)
    truth = np.load(f'optvalues_{nstars}_linearA.npy', allow_pickle=True)

    from pyro_torch_model_5D_linearA import gen_model
    model = gen_model()
    model.eval()
    model.requires_grad_(False)
    #hack to get sample working, log_probs needs to be called first
    #foo = torch.zeros(1, 5, device=device)

    sample_fn, density_fn = get_transforms(model)

    #model.log_probs(foo)
    numdatasamples = 1000000
    ss = pickle.load(open(f'transform_nsamples{numdatasamples}.pkl','rb'))
    #ss = pickle.load(open('transform_cyle_gauss.pkl', 'rb')) #'transform.pkl','rb'))
    nsamples = 100000
    try:
        samples = pickle.load(open(f'nfsamples_nsamples{numdatasamples}.pkl', 'rb'))
    except FileNotFoundError:  
        npts=500
        memory = 100
        z = torch.randn(npts * npts, 5).type(torch.float64).to(device)
        zk = []
        inds = torch.arange(0, z.shape[0]).to(torch.int64)
        for ii in torch.split(inds, int(memory**2)):
            zk.append(sample_fn(z[ii]))
        zk = torch.cat(zk, 0).detach().numpy()
        #samples = ss.inverse_transform(model.sample(num_samples=nsamples).detach().numpy())
        samples = ss.inverse_transform(zk)
        pickle.dump(samples, open(f'nfsamples_nsamples{nsamplesdata}.pkl','wb'))
    plot_cmd(data, samples)

    np.random.seed(222)
    torch.random

    from scipy.stats import norm



    """
    for i in range(10):

        names = ['lnp', 'A', 'gradA', 'sigA', 'c1', 'gradc1', 'sigc1', 'c2', 'gradc2', 'sigc2', 'c3', 'gradc3', 'sigc3', 'c4', 'gradc4', 'sigc4', 'M', 'gradM', 'sigM', 'd', 'gradd', 'sigd']
        data = np.genfromtxt(f'gradients_{i}.txt', names=names) 
        fig, axes = plt.subplots(len(names), figsize=(5, 20))
        for n, ax in zip(names, axes):
            ax.plot(data[n])
            ax.set_ylabel(n)
        plt.tight_layout()
        fig.savefig(f'gradient_t_{i}.pdf', rasterized=True)
        plt.close(fig)

        params = ['A', 'c1', 'c2', 'c3', 'c4', 'M', 'd']
        true =  [truth['A_true'][i], truth['c_true'][i, 0], truth['c_true'][i,1], truth['c_true'][i,2], truth['c_true'][i,3], truth['M_true'][i], truth['d_true'][i]]
        infer = [truth['A'][i],      truth['c'][i, 0],      truth['c'][i,1],      truth['c'][i,2],      truth['c'][i,3],      truth['M'][i],      truth['d'][i]]
        obs   = [None,               truth['c_obs'][i,0],   truth['c_obs'][i,1],  truth['c_obs'][i,2],  truth['c_obs'][i,3],  truth['m_obs'][i],  truth['varpi_obs'][i]]
        obs_sig   = [None,           truth['c_sig'][i, 0],  truth['c_sig'][i,1],  truth['c_sig'][i,2],  truth['c_sig'][i,3],  truth['m_sig'][i],  truth['varpi_sig'][i]]
        dustco = [None, 0.6053750118778156, 0.1073133864066465, 0.06516423983305598, 0.05847777019709274, 0.8958421329783514, None]
        grads = [n for n in names if 'grad' in n]
        sigs = [n for n in names if 'sig' in n]

        
        plt.clf()
        fig, axesall =plt.subplots(len(grads)*2, 2, figsize=(10, 20))
        axes = axesall[[1,3,5,7,9,11,13]]
        ii = 0
        for ax, p, g, sig, t, o, os, dc in zip(axes, params, grads, sigs, true, obs, obs_sig, dustco): 
            if ii ==0: colors = data['lnp'] #np.array(['k']*len(data[p]))
            elif ii < len(params)-2: colors = np.array([norm.logpdf(o, loc=d + a*dc, scale=os) for d, a in zip(data[p], data['A'])])
            elif ii == len(params)-2: colors = np.array([norm.logpdf(o, loc=m - 5*np.log10(d*1e3/10) - a*dc, scale=os) for m, d, a in zip(data[p], data['d'], data['A'])])
            elif ii == len(params)-1: colors = np.array([norm.logpdf(o, loc=1/d, scale=os) for d in data['d']])
            else: print('something is wrong')
            posg = data[g] > 0
            ax[0].scatter(data[p][-1], data[g][-1], color='red', s=200)
            ax[0].scatter(data[p][0], data[g][0], color='grey', s=100)
            ax[0].scatter(data[p][posg], data[g][posg], c=colors[posg], cmap=plt.get_cmap('Blues'))
            #print(np.shape(data[p][~posg]), np.shape(data[g][~posg]), np.shape(colors[~posg]))
            ax[0].scatter(data[p][~posg], data[g][~posg], c=colors[~posg], linewidths=2, cmap=plt.get_cmap('Blues')) #facecolor='none', edgecolors=colors[~posg],
            ax[0].axvline(t, lw=2)
            ax[0].set_xlabel(p)
            ax[0].set_ylabel(g)
            
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax[1].scatter(data[p], data[sig], c=colors, cmap=plt.get_cmap('Blues'))
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax[1].scatter(data[p][-1], data[sig][-1], color='red', s=200)
            ax[1].scatter(data[p][0], data[sig][0], color='grey', s=100)
            ax[1].axvline(t, lw=2)
            ax[1].set_xlabel(p)
            ax[1].set_ylabel(sig)

            xlim0 = ax[0].get_xlim()
            xlim1 = ax[1].get_xlim()
            xlim = [np.min([xlim0[0], xlim1[0]]), np.max([xlim0[1], xlim1[1]])]
            ax[0].set_xlim(xlim)
            ax[1].set_xlim(xlim)
            ii += 1
            #print(data[sig][-2])
        axesprior = axesall[[0,2,4,6,8,10,12]]
        for ax, axp in zip(axes[0], axesprior[0]):
            xlim = ax.get_xlim()
            bins = np.linspace(xlim[0], xlim[1], 10)
            axp.hist(np.random.randn(nsamples), bins=bins, histtype='step', weights=np.ones(nsamples)/nsamples)
            axp.set_xlim(xlim)
            axp.set_xlabel('A')
        for ax, axp in zip(axes[-1], axesprior[-1]):
            o, os = obs[-1], obs_sig[-1]
            xlim = ax.get_xlim()
            bins = np.linspace(xlim[0], xlim[1], 10)
            axp.hist(np.random.randn(nsamples), bins=bins, histtype='step', weights=np.ones(nsamples)/nsamples, label='prior')
            axp.set_xlabel('d')
            axp.legend()


        for ii, (ax, axp, p) in enumerate(zip(axes[1:-1], axesprior[1:-1], params[1:-1])):
            for a, ap in zip(ax, axp):
                xlim = a.get_xlim()
                bins = np.linspace(xlim[0], xlim[1], 10)
                ap.hist(samples[:, ii], bins=bins, histtype='step', weights=np.ones(nsamples)/nsamples, label='prior')
                ap.set_xlim(xlim)
                ap.set_xlabel(p)
                ap.legend()
        plt.tight_layout()
        fig.savefig(f'gradient_param_{i}.pdf', rasterized=True)
        plt.close(fig)
        


        plt.clf()
        fig, axesall =plt.subplots(len(grads)*2, 2, figsize=(10, 20))
        axes = axesall[[1,3,5,7,9,11,13]]
        ii = 0
        for ax, p, g, sig, t, o, os, dc in zip(axes, params, grads, sigs, true, obs, obs_sig, dustco): 
            if ii ==0: colors = data['lnp'] #np.array(['k']*len(data[p]))

            posg = data[g] > 0
            x = data[p]
            y = data['lnp']
            colors = data[g]
            ax[0].scatter(x[-1], y[-1], color='red', s=200)
            ax[0].scatter(x[0], y[0], color='grey', s=100)
            ax[0].scatter(x, y, c=colors, cmap=plt.get_cmap('Blues'))
            #print(np.shape(data[p][~posg]), np.shape(data[g][~posg]), np.shape(colors[~posg]))
            #ax[0].scatter(x[~posg], y[~posg], c=colors[~posg], linewidths=2, cmap=plt.get_cmap('Blues')) #facecolor='none', edgecolors=colors[~posg],
            ax[0].axvline(t, lw=2)
            ax[0].set_xlabel(p)
            ax[0].set_ylabel('lnp')
            try: ymin = np.min(y[y > -9000]); ymax = np.max(y[y > -9000])
            except ValueError: ymin = np.min(y); ymax=np.max(y)
            ax[0].set_ylim(ymin, ymax)
            #breakpoint()
            
            if ii == 0: pass
            elif ii < len(params)-2: y = np.array([norm.logpdf(o, loc=d + a*dc, scale=os) for d, a in zip(data[p], data['A'])])
            elif ii == len(params)-2: y = np.array([norm.logpdf(o, loc=m + 5*np.log10(d*1e3/10) + a*dc, scale=os) for m, d, a in zip(data[p], data['d'], data['A'])])
            elif ii == len(params)-1: y = np.array([norm.logpdf(o, loc=1/d, scale=os) for d in data['d']])
            else: print('something is wrong')
            colors = data[g]
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax[1].scatter(x, y, c=colors, cmap=plt.get_cmap('Blues'))
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax[1].scatter(x[-1], y[-1], color='red', s=200)
            ax[1].scatter(x[0], y[0], color='grey', s=100)
            ax[1].axvline(t, lw=2)
            ax[1].set_xlabel(p)
            ax[1].set_ylabel('lnlike')
        
            try: ymin = np.min(y[y > -9000])
            except ValueError: ymin = np.min(y)
            ax[1].set_ylim(ymin, np.max(y))

            #xlim0 = ax[0].get_xlim()
            #xlim1 = ax[1].get_xlim()
            #xlim = [np.min([xlim0[0], xlim1[0]]), np.max([xlim0[1], xlim1[1]])]
            #ax[0].set_xlim(xlim)
            #ax[1].set_xlim(xlim)
            ii += 1
            #print(data[sig][-2])
        axesprior = axesall[[0,2,4,6,8,10,12]]
        for ax, axp in zip(axes[0], axesprior[0]):
            xlim = ax.get_xlim()
            bins = np.linspace(xlim[0], xlim[1], 10)
            axp.hist(np.random.randn(nsamples), bins=bins, histtype='step', weights=np.ones(nsamples)/nsamples)
            axp.set_xlim(xlim)
            axp.set_xlabel('A')
        for ax, axp in zip(axes[-1], axesprior[-1]):
            o, os = obs[-1], obs_sig[-1]
            xlim = ax.get_xlim()
            bins = np.linspace(xlim[0], xlim[1], 10)
            axp.hist(np.random.randn(nsamples), bins=bins, histtype='step', weights=np.ones(nsamples)/nsamples, label='prior')
            axp.set_xlabel('d')
            axp.legend()


        for ii, (ax, axp, p) in enumerate(zip(axes[1:-1], axesprior[1:-1], params[1:-1])):
            for a, ap in zip(ax, axp):
                xlim = a.get_xlim()
                bins = np.linspace(xlim[0], xlim[1], 10)
                ap.hist(samples[:, ii], bins=bins, histtype='step', weights=np.ones(nsamples)/nsamples, label='prior')
                ap.set_xlim(xlim)
                ap.set_xlabel(p)
                ap.legend()
        plt.tight_layout()
        fig.savefig(f'like_param_{i}.pdf', rasterized=True)
        plt.close(fig)
        """