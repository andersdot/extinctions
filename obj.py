import torch
from scipy import optimize
import torch.nn.functional as F
import math
import numpy as np
from functools import reduce
from collections import OrderedDict
from hessian import hessian 

class PyTorchObjective(object):
    """PyTorch objective function, wrapped to be called by scipy.optimize."""
    def __init__(self, obj_module, filename_grad):
        self.f = obj_module # some pytorch module, that produces a scalar loss
        # make an x0 from the parameters in this module
        parameters = OrderedDict(obj_module.named_parameters())
        self.param_shapes = {n:parameters[n].size() for n in parameters}
        # ravel and concatenate all parameters to make x0
        self.x0 = np.concatenate([parameters[n].data.numpy().ravel() 
                                   for n in parameters])
        self.filename_grad = filename_grad

    def unpack_parameters(self, x):
        """optimize.minimize will supply 1D array, chop it up for each parameter."""
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            #if 'model' in n: n = n[6:]
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters

    def pack_grads(self):
        """pack all the gradients from the parameters in the module into a
        numpy array."""
        grads = []
        for p in self.f.parameters():
            #import pdb; pdb.set_trace()
            grad = p.grad.data.numpy()
            grads.append(grad.ravel())
        return np.concatenate(grads)
    
    def pack_hessian(self):
        return hessian(torch.exp(self.f.logjoint()), self.f.parameters())
        

    def is_new(self, x):
        # if this is the first thing we've seen
        if not hasattr(self, 'cached_x'):
            return True
        else:
            # compare x to cached_x to determine if we've been given a new input
            x, self.cached_x = np.array(x), np.array(self.cached_x)
            error = np.abs(x - self.cached_x)
            return error.max() > 1e-8

    def cache(self, x):
        # unpack x and load into module 
        state_dict = self.unpack_parameters(x)
        self.f.load_state_dict(state_dict)
        # store the raw array as well
        self.cached_x = x
        # zero the gradient
        self.f.zero_grad()
        # use it to calculate the objective
        obj = self.f()
        # backprop the objective
        obj.backward()
        self.cached_f = obj.item()
        self.cached_jac = self.pack_grads()
        self.cached_hes = hessian(self.f(), self.f.parameters())

    def fun(self, x):
        if self.is_new(x):
            self.cache(x)
        parms = np.concatenate([param.data.numpy().ravel() for param in self.f.parameters()], axis=0)
        #print(f'{parms[0].item():8.2f}', f'{self.f.logjoint():8.2f}')
        return self.cached_f

    def jac(self, x):
        if self.is_new(x):
            self.cache(x)
        #print(torch.autograd.grad(self.f.forward(), self.f.parameters()))
        #parms = np.concatenate([param.data.numpy().ravel() for param in self.f.parameters()], axis=0)
        #print(parms)
        #hes = hessian(self.f(), self.f.parameters())
        #sig2 = np.linalg.inv(-1.*self.cached_hes)
        #print(f'{parms[0].item():8.2f}', f'{self.f.logjoint():8.2f}')
        #with open(self.filename_grad, 'a+') as f:
        #    print(f'{self.f.logjoint():8.2f} ', 
        #    f'{parms[0].item():8.2f} ', f'{self.cached_jac[0]:8.2f} ', f'{np.sqrt(sig2[0,0]):8.2f} ', 
        #    f'{parms[1].item():8.2f} ', f'{self.cached_jac[1]:8.2f} ', f'{np.sqrt(sig2[1,1]):8.2f} ', 
        #    f'{parms[2].item():8.2f} ', f'{self.cached_jac[2]:8.2f} ', f'{np.sqrt(sig2[2,2]):8.2f} ',
        #    f'{parms[3].item():8.2f} ', f'{self.cached_jac[3]:8.2f} ', f'{np.sqrt(sig2[3,3]):8.2f} ', 
        #    f'{parms[4].item():8.2f} ', f'{self.cached_jac[4]:8.2f} ', f'{np.sqrt(sig2[4,4]):8.2f} ', 
        #    f'{parms[5].item():8.2f} ', f'{self.cached_jac[5]:8.2f} ', f'{np.sqrt(sig2[5,5]):8.2f} ', 
        #    f'{parms[6].item():8.2f} ', f'{self.cached_jac[6]:8.2f} ', f'{np.sqrt(sig2[6,6]):8.2f} ', file=f)
        return self.cached_jac
    
    def hes(self, x):
        if self.is_new(x):
            self.cache(x)
        sig2 = np.linalg.inv(-1.*self.cached_hes)
        diag = [sig2[i,i] for i in range(sig2.shape[0])]
        print([f'{d:8.2f} ' for d in diag])
        return self.cached_hes
