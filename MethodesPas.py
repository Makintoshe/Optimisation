import numpy as np
from scipy.optimize import golden

"""
    d: search direction
    x: initial point
    cost: cost function
    gradient: gradient function
"""
def backtrack(x, d, cost, gradient, sigma0=1, c=.1, rho=.5, iter_max=1000):
    assert x.shape == d.shape
    sigma = sigma0    
    crit = gradient(x).T @ d
    iter_sigma = 0
    while cost(x + sigma * d) or (cost(x + sigma * d) > cost(x) + c * sigma * crit and iter_sigma < iter_max):
        sigma *= rho
        iter_sigma += 1
    if np.isnan(cost(x + sigma * d)):
        raise AssertionError("Pas de sigma pour iter_max = " + iter_max)
    return sigma, iter_sigma

"""
    P: P of quadratic problem
"""
def Quadratic_optimal_step(x,grad, P):
    #num = grad.T@grad
    #denom = grad.T@P@grad
    return grad.T@grad/grad.T@P@grad

"""
    a : lower bound of the interval
    b : upper bound of the interval
    ε : précision
"""
def dichotomie(f, a, b, ε=1e-8):
    while abs(a - b) > ε:
        m = (a + b) / 2
        if f(m) > f(a) :
            b = m
        else :
            a = m
    return f((a + b) / 2)

"""
    Return the minimum of a function of one variable using golden section method. Source : vu sur internet.
"""
def golden_search(cost,a,b,c,precision):
    if abs(a - b) < precision:
        return (a + b)/2
    # Create a new possible center, in the area between c and b, pushed against c
    d = c + resphi*(b - c)
    if cost(d) < cost(c):
        return golden_search(cost, c, d, b, precision)
    else:
        return golden_search(cost, d, c, a, precision)
    
    
"""
    Golden Search method of Scipy.Optimize : return the minimum of a function of one variable using golden section method. Source : vu sur internet.
"""
def golden_search_by_scipy(cost):
    return golden(cost)