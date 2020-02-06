import numpy as np
from scipy.special import gammaln, gamma


def log_dirichlet(dist, hyper):
    """
    dist: distribution as a numpy np.array
    hyper: Dirichlet parameters, base measure times concentration parameter
    """
    assert len(dist)==len(hyper)
    dist = np.array(dist); hyper = np.array(hyper)
    return gammaln(np.sum(hyper))-np.sum(gammaln(hyper))+np.sum((hyper-1)*np.log(dist))


def log_beta(p, hyper):
    return log_dirichlet(np.array((p, 1-p)), hyper)


def log_polya(N, A):
    """
    N: Counts as a numpy np.array
    A: Dirichlet parameters, base measure times concentration parameter
    """

    ret = gammaln(A.sum()) - gammaln((N + A).sum()) + (gammaln(N + A)).sum() - (gammaln(A)).sum()
    if np.isnan(ret):
        raise Exception('Dirichlet base measure cannot be zero. A=%s' % (A))
    
    return ret


def non_log_dirichlet(dist, hyper):
    assert len(dist)==len(hyper)
    return (gamma(np.sum(hyper))*np.prod(dist**(hyper-1)))/np.prod(gamma(hyper))


def log_gamma(x, shape, scale):
    x = np.asarray(x)
    assert (x > 0).all and shape > 0 and scale > 0 
    return -shape*np.log(scale) - gammaln(shape) + (shape-1)*np.log(x) - x/scale


def log_gamma_poisson(N, shape, rate):
    """
    N: Counts as a numpy np.array
    shape, rate: Parameters of the gamma distribution
    
    """
    sum_N_plus_shape = (N+shape).sum() #this is computation is performed twice
    return gammaln(sum_N_plus_shape) - gammaln(rate) + np.log(shape*rate) - np.log( (len(N) + rate)*(sum_N_plus_shape)) - (gammaln(N+1)).sum()


def main():
    """
    alpha_sym = np.array([10,10,10,10,10])
    alpha_asym = np.array([10,100,35,1000,5])
    N=10000
    samples = dirichlet(alpha_sym, N)
    log_likelihood = [log_dirichlet(s, alpha_sym) for s in samples]
    likelihood = [non_log_dirichlet(s, alpha_sym) for s in samples]
    plt.hist(likelihood, 1000, facecolor='green')
    plt.show()
    """
    N = 1000
    shape = 10.0
    scale = 1.0
    samples = gamma(shape, scale, N)
    np.log_likelihood = [log_gamma(s, shape, scale) for s in samples]
    

if __name__ == "__main__":
    main()

