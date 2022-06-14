import numpy as np
import MethodesPas as mp


"""    
    P: P matrix of Quadratic problem (if backtrack then P is note using -> P=None)
    cost: cost function
    x0: initial point
    gradient: gradient function call
    hessian: hessian func 
    flist : list of evaluate points with cost function
    nlist : liste of gradient norms
    xlist : list of points
    
"""
def steepest_descent(x0,cost,gradient, method, P=None, sigma0=1, c=.1, rho=.5, iter_max=1000, ε=1e-8):
    
    xlist, flist, nlist = [x0],[cost(x0)],[np.linalg.norm(gradient(x0))]
    k = 0
    while np.linalg.norm(gradient(xlist[k])) > ε and k < iter_max:
        d = - gradient(xlist[k])
        sigma = sigma0
        if method == mp.backtrack:
            sigma = method(xlist[k], d, cost, gradient, sigma, c , rho, iter_max)   
        else:
            sigma = method(xlist[k],gradient(xlist[k]), P)   
        xlist.append(xlist[k] + sigma[0] * d)
        flist.append(cost(xlist[k]))
        nlist.append(np.linalg.norm(gradient(xlist[k])))
        k  = k + 1
    return xlist,flist,nlist

"""    
    method : research step function
    P: P matrix of Quadratic problem (if backtrack then P is note using -> P=None)
    cost: cost function
    x0: initial point
    gradient: gradient function call
    hessian: hessian func 
    flist : list of evaluate points with cost function
    nlist : liste of gradient norms
    xlist : list of points
    
"""
def steepest_descent_norm_l1(x0, cost, gradient, method, P=None, sigma0=1, c=.1, rho=.5, iter_max=1000, ε=1e-8):

    xlist, flist, nlist = [x0],[cost(x0)],[np.linalg.norm(gradient(x0))]
    k = 0
    while np.linalg.norm(gradient(xlist[k])) > ε and k < iter_max:
        ll = list(np.abs(gradient(xlist[k])))
        index =  ll.index(np.max(ll))
        d = np.zeros(np.shape(gradient(xlist[k])))
        d[index] = gradient(xlist[k])[index]
        d *= (-1) 
        sigma = sigma0
        if method == fp.backtrack:
            sigma = method(xlist[k], d, cost, gradient, sigma, c , rho, iter_max)   
        else:
            sigma = method(xlist[k],gradient(xlist[k]), P)
        xlist.append(xlist[k] + sigma[0] * d)
        flist.append(cost(xlist[k]))
        nlist.append(np.linalg.norm(gradient(xlist[k])))
        k = k + 1
    return xlist,flist,nlist


"""    
    cost: cost function
    x0: initial point
    gradient: gradient function call
    hessian: hessian func 
    flist : list of evaluate points with cost function
    nlist : liste of gradient norms
    xlist : list of points
    
"""
def Newton_descent(x0,cost,gradient,hessian, iter_max=1000, ε=1e-8):
    
    xlist, flist, nlist = [x0],[cost(x0)],[np.linalg.norm(gradient(x0))]
    k = 0
    while np.linalg.norm(gradient(xlist[k])) > ε and k < iter_max:
        xlist.append(xlist[k] - np.linalg.inv(hessian(xlist[k])) @ gradient(xlist[k]))
        flist.append(cost(xlist[k]))
        nlist.append(np.linalg.norm(gradient(xlist[k])))
        k  = k + 1
    return xlist,flist,nlist

"""
    cost: cost function
    x0: initial point
    gradient: gradient function call
    flist : list of evaluate points with cost function
    nlist : liste of gradient norms
    xlist : list of points
    
    REMARK : i don't use golden search directly!!!
"""
def coordinate_descent(x0,cost,gradient,min_1d, sigma0=1, iter_max=1000, ε=1e-8): 
    
    xlist, flist, nlist = [x0],[cost(x0)],[np.linalg.norm(gradient(x0))]
    k = 0
    sigma = sigma0
    while True:
        grad = gradient(xlist[k])
        x = np.copy(xlist[k])
        for i in range(len(gradient(xlist[k]))):
            if grad[i] > 0:
                interval = [xlist[k][i] - sigma, xlist[k][i]]
            else :
                interval = [xlist[k][i], xlist[k][i] + sigma]
            sigma = min_1d(cost, interval)
            x[i] = xlist[k][i] - sigma * grad[i]
        xlist.append(x)
        flist.append(cost(x))
        nlist.append(np.linalg.norm(gradient(x)))
        k += 1
        if cost(xlist[k]) - cost(xlist[k - 1]) < ε or k < iter_max :
            break
    return xlist,flist,nlist