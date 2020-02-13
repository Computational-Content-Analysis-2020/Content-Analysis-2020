'''
Created on Jun 14, 2013

@author: Juston Moore

Partially based on Ryan Prescott Adams' code at http://machinelearningmisc.blogspot.com/2010/01/slice-sampling-in-python.html
'''

import numpy as np
from math import floor

def slice_sample(x_init, ll_func, window_size, L_bound=None, R_bound=None, step_out_limit=float('inf')):
    """
    x_init: current value of the parameter to be sampled
    ll_func: function that takes x as a parameter and returns a log-prob for
        the parameter value
    L_bound: Lower value on the support of the likelihood function.
        None if support is unbounded
    R_bound: Upper value on the support of the likelihood function.
        None if support is unbounded
    step_out_limit: UN-IMPLEMENTED Limit on the number of step-out iterations
        (Neal 2003, Algorithm in Fig. 3)
        0 for no step-out, float('inf') for unlimited step-out
    """
    global ITERATIONS_TAKEN
    ITERATIONS_TAKEN = 0
    
    # Define the vertical slice
    z_init = ll_func(x_init)
    z = z_init - np.random.exponential()
    
    # Define the "window" around x_init
    L = x_init - window_size * np.random.rand()
    R = L + window_size
    if L_bound is not None:
        L = max(L_bound, L)
    if R_bound is not None:
        R = min(R_bound, R)

    # print ""
    # print "* x=", x_init
    # print "* L=", L, "L_bound=", L_bound
    # print "* R=", R, "R_bound=", R_bound

    
    # Do the stepping-out procedure
    if np.isinf(step_out_limit):
        L_step_out_limit = float('inf')
        R_step_out_limit = float('inf')
    elif step_out_limit <= 0:
        L_step_out_limit = 0
        R_step_out_limit = 0
    else:
        L_step_out_limit = floor(np.random.rand() * step_out_limit)
        R_step_out_limit = step_out_limit - L_step_out_limit
        
    while L_step_out_limit > 0 and (L_bound is None or L_bound < L) and ll_func(L) > z:
        L -= window_size
        if L_bound is not None:
            L = max(L_bound, L)

        # print "  L=", L, "L_bound=", L_bound
        L_step_out_limit -= 1

    while R_step_out_limit > 0 and (R_bound is None or R < R_bound) and ll_func(R) > z:
        R += window_size
        if R_bound is not None:
            R = min(R_bound, R)
        
        # print "  R=", R, "R_bound=", R_bound
        R_step_out_limit -= 1
    
    assert L <= x_init and x_init <= R, 'L = %s, R = %s, L_bound = %s, R_bound = %s, x_init=%s' % (L, R, L_bound, R_bound, x_init) 
        
    # print 'L = %s, R = %s, L_bound = %s, R_bound = %s, x_init=%s' % (L, R, L_bound, R_bound, x_init) 
        
    while True:
        ITERATIONS_TAKEN += 1
        
        # Note: we have to allow the new x to be exactly the old x because of
        # discrete floating point representation.
        assert (R-L) >= 0, "R = %.4f, L=%.4f" % (R, L)
        x = L + np.random.rand() * (R-L)
        
        if ll_func(x) >= z:
            ## TODO: remove this assert
            # assert x > L_bound and (R_bound is None or x < R_bound)
            return x
        else:
            if x < x_init:
                L = x
            else:
                R = x

def multivariate_slice_sample(x_init, ll_func, window_size, L_bound=None, R_bound=None, gradient_func=None):
    """
    Multivariate slice sampler using the hyper-rectangle method
    """
    
    x_init = np.array(x_init, dtype=float)
    window_size = np.zeros(x_init.shape) + window_size
    assert x_init.shape == window_size.shape
    
    z_init = ll_func(x_init.copy())
    z = z_init - np.random.exponential()

    L = x_init - np.random.rand(*x_init.shape) * window_size
    R = L + window_size
    if L_bound is not None:
        L = np.maximum(L_bound, L)
    if R_bound is not None:
        R = np.minimum(R_bound, R)
    
    while True:
        assert ((R-L) > 0).all()

        ## check when testing
        # check = ll_func(x_init.copy())
        # if z_init != check:
        #     print z_init, check
        #     assert 0 == 1
        
        x = L + np.random.rand(*x_init.shape) * (R-L)
        
        if ll_func(x.copy()) >= z:
            return x
        else:
            if gradient_func is not None:
                dim = ((R-L) * np.abs(gradient_func(x))).argmax()
                if x[dim] < x_init[dim]:
                    L[dim] = x[dim]
                else:
                    R[dim] = x[dim]
            else:
                lmask = x < x_init
                L[lmask] = x[lmask]
                rmask = (lmask == False)
                R[rmask] = x[rmask]

def shrinking_rank_slice_sample(x_init, ll_func, gradient_func, crumb_covariance, L_bound=None, R_bound=None, shrink_rate = 0.9):
    """
    The covariance-adaptive Shrinking-Rank slice sampler from Madeleine B. Thompson's thesis, Figure 3.7, p. 61
    sigma_c: crumb standard deviation for initial, uniform proposal
    shrink_rate: crumb covariance decay rate
    """
    
    def projection(v):
        if subspace_columns == 0:
            return v
        else:
            J_subspace = J[:,:subspace_columns]
            return v - np.dot(np.dot(J_subspace, J_subspace.T), v)
    
    x_init = np.asarray(x_init, dtype=float)
    p = len(x_init)
    assert x_init.shape == (p,)
    
    z_init = ll_func(x_init)
    z = z_init - np.random.exponential()
    
    subspace_columns = 0
    J = np.zeros((p, p-1))
    
    inverse_sample_covariance = 0
    unscaled_sample_mean = 0
    
    while True:
        crumb = projection(np.random.multivariate_normal(x_init, crumb_covariance * np.identity(p)))
        
        inverse_sample_covariance += 1. / crumb_covariance
        sample_covariance = 1. / inverse_sample_covariance
        unscaled_sample_mean += 1. / crumb_covariance * (crumb - x_init)
        sample_mean = sample_covariance * unscaled_sample_mean
        
        # NOTE: There is an error in the thesis' algorithm here. sigma_x should be squared. See correct implementation at http://www.cs.utoronto.ca/~radford/ftp/cass.r
        x = x_init + projection(np.random.multivariate_normal(sample_mean, sample_covariance * np.identity(p)))
        
        if (L_bound is not None and (x <= L_bound).any()) \
            or (R_bound is not None and (x >= R_bound.any())):
            #crumb_covariance *= 0.1
            continue
        elif ll_func(x) >= z:
            return x
        else:
            gradient = gradient_func(x)
            projected_gradient = projection(gradient)
            
            if subspace_columns < p-1:
                normed_gradient = gradient / np.sqrt(np.dot(gradient, gradient))
                normed_projected_gradient = projected_gradient / np.sqrt(np.dot(projected_gradient, projected_gradient))
                
                if np.dot(normed_projected_gradient, normed_gradient) > 0.5:
                    J[:,subspace_columns] = normed_projected_gradient
                    subspace_columns += 1
                    continue
        
            crumb_covariance *= shrink_rate
