#!/usr/bin/python

# multivariate Hawkes process -- simulation and Bayesian inference
# Richard Kwo
# May, 2014

from numpy import zeros, ones, identity, bincount, log, exp, abs, sqrt, savez, savetxt, shape, eye, all, any, argmin, argmax, array, mean, linspace, sum, loadtxt, concatenate, amax, diag
from slice_sampler import slice_sample
from numpy.random import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.special import gammaln

from likelihoods import log_gamma

from scipy.optimize import minimize


# class begin
class Hawkes(object):
    """docstring for Hawkes"""
    def __init__(self, A=2, T=100.0, non_diagonal=False):
        '''
        Hawkes object

        A: number of agents

        T: max time 

        non_diagonal: if True, the diagonal of the nu matrix is enforced to zero 
        '''
        # TODO: the name ``non_diagonal`` is confusing 
        assert A>=1

        self._A = A
        self._T = T * 1.0
        self._non_diagonal = non_diagonal

        print ("Hawkes initialized (A=%d, T=%.2f, non_diagonal=%s)" % (self._A, self._T, self._non_diagonal))

        # parameters        
        self._nu = zeros((self._A, self._A))
        self._tau = zeros(self._A)
        self._baseline_rate = zeros(self._A)

        # hyperparams
        self._hyperparams_tau = None    # gamma (shape, scale) hyperparams

        self._nu_set = False
        self._tau_set = False
        self._baseline_rate_set = False

        # data 
        self._data = None

        # locks
        self._ll_cached = False
        self._prior_cached = False
        self._log_likelihood = None
        self._log_likelihood_test_set = None
        self._log_prior = None



    def get_A(self):
        ''' get A '''
        return self._A

    def get_T(self):
        ''' get max time T'''
        return self._T

    def set_nu(self, nu, x=None, y=None):
        '''set the jump height matrix self._nu[x,y] to nu.

        When setting the whole self._nu to an A x A matrix nu, passing x = None and y = None

        nu[p, q] is the excitation height from p to q

        '''
        if all(self._nu[x,y] == nu): return  # avoid reevaluating ll when not changed
        if x is None and y is None:
            assert shape(nu) == (self._A, self._A) and all(nu>=0), '%s' % (nu)
            if self._non_diagonal:
                assert all(diag(nu)==0), "%s" % (nu)
        else:
            assert nu>=0
            if self._non_diagonal:
                assert x!= y or nu==0
        
        self._nu[x,y] = nu * 1.0
        self._nu_set = True
        # cache lock
        self._ll_cached = False

    def set_tau(self, tau, x=None):
        '''set the time decays, of shape (A)'''
        if all(self._tau[x] == tau): return  # avoid reevaluating ll when not changed
        if x is None:
            assert shape(tau) == (self._A, ) and all(tau>0), '%s' % (tau)
        else:
            assert tau>0

        self._tau[x] = tau * 1.0
        self._tau_set = True

        # cache lock
        self._ll_cached = False


    def set_baseline_rate(self, baseline_rate, x=None):
        '''set the baseline_rate (lambda_0), of shape(A)'''
        if all(self._baseline_rate[x] == baseline_rate): return  # avoid reevaluating ll when not changed
        if x is None:
            assert shape(baseline_rate) == (self._A, ) and all(baseline_rate>0), '%s' % (baseline_rate)
        else:
            assert baseline_rate>0

        self._baseline_rate[x] = baseline_rate * 1.0
        self._baseline_rate_set = True

        # cache lock
        self._ll_cached = False

    def get_nu(self):
        return self._nu.copy()

    def get_tau(self):
        return self._tau.copy()

    def get_baseline_rate(self):
        return self._baseline_rate.copy()

    def test_stationarity(self):
        time_decay_mat = zeros((self._A, self._A))
        for x in range(self._A):
            time_decay_mat[:,x] = self._tau
        gamma_mat = self._nu * time_decay_mat
        w, v = np.linalg.eig(gamma_mat)
        if max(abs(w))<1:
            return True
        else:
            # print "Not stationary: sprectrum is", w
            return False

    def test_matrix_norm_constraint(self):
        '''
        Test if the matrix norm constraint (1->1 matrix norm < 1, 
            i.e. all the col sum of the matrix[x,y] = nu[x,y] * tau[y] < 1) is satisfied. 
        
        This is a stronger condition to ensure stationarity.
        '''
        colsum = sum(self._nu, axis=0) * self._tau
        if all(colsum<1): 
            return True
        else:
            return False


    def plot_data_events(self, show_figure=True):
        '''
        Show a vertical line plot of data.
        '''
        assert self._data is not None

        fig = plt.figure()
        for x in range(self._A):
            ax = fig.add_subplot(self._A,1,x+1)
            ax.vlines(x=self._data[x], ymin=0, ymax=1)
            ax.axvline(x=self._time_to_split, ls='--', color='r')
            ax.set_title("agent %d (%d events)" % (x+1, len(self._data[x])))
            ax.set_xlim(0, self._T)
            ax.set_ylim(0, 1)

        plt.tight_layout(pad=1, h_pad=1, w_pad=1)
        if show_figure:
            plt.show()


    def simulate(self):
        '''simulate a sample path of the multivariate Hawkes process 

        method: Dassios, A., & Zhao, H. (2013). Exact simulation of Hawkes process with exponentially decaying intensity. Electronic Communications in Probability

        ''' 
        assert self._nu_set and self._tau_set and self._baseline_rate_set, "parameters not set yet"
        assert self.test_stationarity(), "This is not a stationary process!"

        currentTime = 0
        # data[x] is the time points for x
        data = [[] for x in range(self._A)]
        # the left and right limit of intensities
        intensities_left = self._baseline_rate.copy()
        intensities_right = self._baseline_rate.copy() + 1e-10
        # event counts
        event_counts = [0 for x in range(self._A)]

        print ("Simulating...")
        iter_count = 0
        while currentTime <= self._T:
            iter_count += 1
            # get W
            s = -ones(self._A)
            for x in range(self._A):
                u1 = random()
                D = 1 + log(u1)/(intensities_right[x] - self._baseline_rate[x])/self._tau[x]
                u2 = random()
                s2 = -1/self._baseline_rate[x] * log(u2)
                if D>0:
                    s1 = -self._tau[x] * log(D)
                    s[x] = min(s1, s2)
                else:
                    s[x] = s2
            assert all(s>=0)
            # event to l 
            l = argmin(s)
            W = s[l]

            # record jump
            currentTime = currentTime + W
            data[l].append(currentTime)
            event_counts[l] += 1

            # update intensities
            intensities_left = (intensities_right - self._baseline_rate) * exp(-W / self._tau) + self._baseline_rate
            intensities_right = intensities_left + self._nu[l,:]

        print ("Simulation done, %d events occurred" % (sum(event_counts)))

        return data

    def loadData(self, data, time_to_split=100.0):
        '''
        Load data into the object. The data can be later used to sample the parameters.

        time_to_split: the time to split the training set (t<=time_to_split) and the test set (t>time_to_split). 

        data: a list. data[x] is the list of event times of agent x.
        '''
        assert len(data)==self._A
        assert time_to_split>0 and time_to_split <= self._T
        self._time_to_split = time_to_split
        self._train_set_sizes = [0 for x in range(self._A)]
        self._test_set_sizes = [0 for x in range(self._A)]
        for x in range(self._A):
            # the times should be sorted
            # no duplicates !!! (very important, otherwise process messed up)
            assert all(data[x][i] < data[x][i+1] for i in range((len(data[x])-1)))

        self._data = data[:]
        for x in range(self._A):
            for t in self._data[x]:
                if t<=time_to_split:
                    self._train_set_sizes[x] += 1
                else:
                    self._test_set_sizes[x] += 1

        print ("\nData loaded into hawkes.")
        print ("Split time set to %.3f" % (self._time_to_split))
        for x in range(self._A):
            print ("-- %d events for agent %d  (train: %4d, test: %4d)" % (len(self._data[x]), x+1, self._train_set_sizes[x], self._test_set_sizes[x]))

        print ("Caching data...")
        self.updateDataCache()

    def getData(self):
        return self._data[:]

    def updateDataCache(self):
        '''
        precompute and cache the elapsed times between two consecutive events to expedite the evaluation of likelihood.

        This cache does not depend on parameters. No need to re-cache after resetting parameters.
        '''
        # _newly_added_elapsed_time[x][k] is the array of elapsed time for events in the range [data[x][k-1], data[x][k])
        self._newly_added_elapsed_time = {}
        # _newly_added_event_from[x][k] is the corresponding indexing array of the rows of _nu
        self._newly_added_event_from = {}
        for x in range(self._A):
            self._newly_added_elapsed_time[x] = []
            self._newly_added_event_from[x] = []
            for k, currentTime in enumerate(self._data[x]):
                tmp_elapsed_times = []
                tmp_event_from = []
                for y in range(self._A):
                    for t in self._data[y]:
                        if  (k==0 or self._data[x][k-1] <= t) and t < currentTime:
                            tmp_elapsed_times.append(currentTime - t)
                            tmp_event_from.append(y)

                # should be only ONE empty array -- corresponding the first event of all events of all x
                self._newly_added_elapsed_time[x].append(array(tmp_elapsed_times))
                self._newly_added_event_from[x].append(array(tmp_event_from, dtype=int))

        print ("Data cache updated.")

    def integratedRate(self, x, t):
        '''
        Return the integral of x's rate function from 0 to t. Should be a continuous function of t.
        '''
        assert t>=0

        t = t * 1.0
        integrated_rate = self._baseline_rate[x] * t
        for q in range(self._A):
            times_before = array([s for s in self._data[q] if s<t])
            tmp = sum(1 - exp(-(t - times_before) / self._tau[x] ) )
            tmp = tmp * self._tau[x] * self._nu[q, x]
            integrated_rate += tmp

        return integrated_rate

    def rate(self, x, t, right_limit=True):
        '''
        Return the rate function of dim x at time t. 

        right_limit: if True, return the value from the right limit; if False, return from the left limit.
        '''
        assert t>=0

        t = t * 1.0
        r = self._baseline_rate[x]
        for q in range(self._A):
            if right_limit:
                times_before = array([s for s in self._data[q] if s <= t])
            else:
                times_before = array([s for s in self._data[q] if s < t])
            if len(times_before)>0:
                r += self._nu[q, x] * sum(exp(-(t - times_before) / self._tau[x]))
                
        return r

    def eventCount(self, x, t):
        '''
        Return N_t of dimension x
        '''
        assert t>=0

        t *= 1.0
        return len([s for s in self._data[x] if s<=t])

    def log_likelihood_test_set(self):
        '''
        Return the log_likelihood of the test set conditioned on the training set data, given the current parameters
        '''
        if self._ll_cached:
            return self._log_likelihood_test_set
        else:
            # re-evaluate and cache it
            self.log_likelihood()
            return self._log_likelihood_test_set

    def set_hyperparams_tau(self, gamma_shape, gamma_scale):
        assert gamma_shape>0 and gamma_scale>0
        self._hyperparams_tau = array([gamma_shape, gamma_scale])

    def log_prior(self):
        if self._prior_cached:
            return self._log_prior
        else:
            assert all(self._hyperparams_tau>0)
            # prior for tau
            self._log_prior = sum(log_gamma(self._tau, *self._hyperparams_tau))
            self._prior_cached = True
            return self._log_prior


    def log_likelihood(self):
        '''
        Evaluate the log likelihood of the data given the parameters (self._nu, self._tau, self._baseline_rate)

        Return: the log_likelihood of the training set (those with t <= time_to_split).

        By running the function, it also updates the cached log_likelihood for the test set (those with t > time_to_split)
        '''
        if self._ll_cached:
            return self._log_likelihood
        else:
            # compute the ll: ll of the train set (0 < t <= time_to_split)
            # ll_test: ll of the test set (time_to_split <t <= T)
            ll = 0
            ll_test = 0
            for x in range(self._A):
                ## ll for each agent
                integrated_rate_T = self.integratedRate(x, self._T)
                integrated_rate_split_time = self.integratedRate(x, self._time_to_split)
                ll -= integrated_rate_split_time
                ll_test -= integrated_rate_T - integrated_rate_split_time
                rate_sum_x = 0
                log_rate_sum_x = 0
                log_rate_sum_x_test = 0
                assert len(self._newly_added_event_from[x])==len(self._data[x])
                train_set_size = 0
                test_set_size = 0
                # both the test set and the train set share the SAME rate function
                # the difference is the range for the log sum
                for k, currentTime in enumerate(self._data[x]):
                    # iterative update
                    ## extra decay of old events since last k, no extra decay for k==0
                    if k>0:
                        rate_sum_x *= exp(-(currentTime - self._data[x][k-1]) / self._tau[x] )
                    ## newly added from the cached data
                    if len(self._newly_added_event_from[x][k])>0:
                        corresponding_nu = self._nu[self._newly_added_event_from[x][k], x]
                        rate_sum_x += sum(corresponding_nu * exp(-self._newly_added_elapsed_time[x][k] / self._tau[x]))
                    # remember to add the non-decaying baseline rate
                    if currentTime <= self._time_to_split:
                        ## train set
                        log_rate_sum_x += log(rate_sum_x + self._baseline_rate[x])
                        train_set_size += 1
                    else:
                        ## test set
                        log_rate_sum_x_test += log(rate_sum_x + self._baseline_rate[x])
                        test_set_size += 1
                # now add the ll for this dim
                ll += log_rate_sum_x 
                ll_test += log_rate_sum_x_test
                
            # ll computed, cache it
            self._log_likelihood = ll
            self._log_likelihood_test_set = ll_test
            self._ll_cached = True
            return self._log_likelihood

    def log_likelihood_slow(self):
        '''
        Gives a slow computed result of the likelihood; no caching is used. 
        Should return the same value as self.log_likelihood()
        '''
        ll = 0
        for x in range(self._A):
            ll -= self.integratedRate(x, self._time_to_split)
            for t in self._data[x]:
                if t<self._time_to_split:
                    ll += log(self.rate(x, t, right_limit=False))
        return ll

    def randomize_parameters(self):
        '''
        Randomize the parameters -- self._nu, self._tau, self._baseline_rate
        '''
        print ("Randomizing parameters to satisfy the matrix norm constraint...")
        while True:
            nu_randomized = np.random.gamma(shape=1, scale=2, size=(self._A, self._A))
            if self._non_diagonal:
                for x in range(self._A):
                    nu_randomized[x,x] = 0            
            self.set_nu(nu_randomized)
            self.set_baseline_rate(np.random.gamma(shape=1, scale=0.2, size=(self._A)))
            if self._hyperparams_tau is not None:
                self.set_tau(np.random.gamma(*self._hyperparams_tau, size=(self._A)))
            else:
                self.set_tau(np.random.gamma(shape=1, scale=1.5, size=(self._A)))

            if self.test_matrix_norm_constraint(): break
        # # cache
        # if self._data is not None:
        #     self.updateDataCache()
        print ("Parameters randomized.")

    def optimize_parameters(self, verbose=0):
        '''
        MLE of the parameters by optimization.

        Return: baseline_rate, tau, nu
        '''
        print ("\n\n")

        def evaluate_ll(params):
            # params = [baseline_rate[0],...,baseline_rate[A-1], tau[0],..., tau[A-1], nu...]

            this_baseline_rate = abs(params[0:self._A])
            this_tau = abs(params[self._A:2*self._A])
            this_nu = abs(params[2*self._A:])
            this_nu.resize((self._A, self._A))
            if self._non_diagonal:
                for x in range(self._A):
                    this_nu[x,x] = 0

            self.set_baseline_rate(this_baseline_rate)
            self.set_tau(this_tau)
            self.set_nu(this_nu)

            return (-self.log_likelihood())


        self.randomize_parameters()
        init_guess = concatenate((self.get_baseline_rate(), self.get_tau(), self.get_nu().reshape((1,-1))[0]))
        print ("Optimizing parameters...")
        # print init_guess
        bounds = []
        for x in range(self._A): bounds.append((1e-4, 30))   # baseline
        for x in range(self._A): bounds.append((0.11, 20))    # tau
        for x in range(self._A):   # nu
            for w in range(self._A):
                if self._non_diagonal and x==w:
                    bounds.append((0, 0))
                else:
                    bounds.append((1e-6, 100))
        
        def cons_lambdas(j):
            return lambda x: 1 - sum(x[2*self._A:].reshape((self._A, self._A))[:,j]) * x[self._A+j] - 1e-6

        cons = []
        for q in range(self._A):
            this_con = {}
            this_con['type'] = 'ineq'
            this_con['fun'] = cons_lambdas(q)
            cons.append(this_con)

        res = minimize(evaluate_ll, init_guess, method="SLSQP", bounds=bounds, 
            constraints=cons, options={'disp': (verbose>0), 'maxiter':1000, 'ftol':1.0e-9})

        if verbose>0:
            print (res)
        assert res.success, res.message

        mle_params = res.x
        baseline_rate = mle_params[0:self._A]
        tau = mle_params[self._A:self._A*2]
        nu = mle_params[2*self._A:]
        nu.resize((self._A, self._A))

        # numerical +/- issue when absolute value is near zero
        baseline_rate = abs(baseline_rate)
        tau = abs(tau)
        nu = abs(nu)
        if self._non_diagonal:
            for x in range(self._A):
                nu[x,x] = 0

        if verbose>0:
            print ("baseline_rate =", baseline_rate)
            print ("tau =", tau)
            print ("nu =", nu)

        # print "cons"
        # for this_con in cons:
        #     print this_con['fun'](mle_params)
        assert self.test_matrix_norm_constraint(), "matrix norm constraint violated! %s" % (sum(nu, axis=0) * tau)

        self._baseline_rate_window_size = amax(baseline_rate)
        self._tau_window_size = amax(tau)
        self._nu_window_size = amax(nu)
        if verbose>0:
            print ("Setting the window size of the slice sampler to be the maximum")
            print ("baseline", self._baseline_rate_window_size, "tau", self._tau_window_size, "nu", self._nu_window_size)

        return baseline_rate, tau, nu

    def sample_tau(self, use_prior=False):
        '''
        Update self._tau of shape (self._A) with one slice sampling draw.

        To ensure stationarity, which suffices to be ensured by tau(x) * nu(y,x) <1 for every (x,y). 
        The upper bound for sampling tau(x) is set to be min(1/nu(y,x)) among y.

        Return the draw of self._tau
        '''
        ## closure
        def sample_func_tau(tau):
            self.set_tau(tau, x)
            if use_prior:
                return self.log_likelihood() + self.log_prior()
            else:
                return self.log_likelihood()

        for x in range(self._A):
            ## constraining spectral radius by the max col sum
            upper_bound_tau = 1.0 / sum(self._nu[:,x])
            slice_sample(self._tau[x], sample_func_tau, window_size=self._tau_window_size, L_bound=0.1, R_bound=upper_bound_tau, step_out_limit=4)

        return self.get_tau()

    def sample_nu(self, use_prior=False):
        '''
        Update self._nu with one slice sampling draw

        To ensure stationarity, the upper bound for sampling nu(x,y) is set to be 1/tau(y).

        Return the draw of self._nu
        '''
        def sample_func_nu(nu):
            self.set_nu(nu, x, y)
            if use_prior:
                return self.log_likelihood() + self.log_prior()
            else:
                return self.log_likelihood()

        for x in range(self._A):
            for y in range(self._A):
                if self._non_diagonal:
                    if x==y: 
                        assert self._nu[x,y]==0
                        continue
                upper_bound_nu = 1.0 / self._tau[y] - (sum(self._nu[:,y]) - self._nu[x,y])
                slice_sample(self._nu[x,y], sample_func_nu, window_size=self._nu_window_size, L_bound=0, R_bound=upper_bound_nu, step_out_limit=4)

        return self.get_nu()

    def sample_tau_and_nu_revised(self, use_prior=False):
        '''
        Update self._tau and self._nu with one slice sampling round

        The ordering is specified as

        for x in range(A):
            sample tau[x]
            for y in range(A):
                sample nu[y, x]

        Return one draw of self._tau, self._nu
        '''
        ## closure
        def sample_func_tau(tau):
            self.set_tau(tau, x)
            if use_prior:
                return self.log_likelihood() + self.log_prior()
            else:
                return self.log_likelihood()

        def sample_func_nu(nu):
            ## !! [y,x] not [x,y] !!
            self.set_nu(nu, y, x)
            if use_prior:
                return self.log_likelihood() + self.log_prior()
            else:
                return self.log_likelihood()        

        for x in range(self._A):
            upper_bound_tau = 1.0 / sum(self._nu[:,x])

            # TODO: remove the assert
            assert self._tau[x] <= upper_bound_tau, "tau = %f, upper_bound_tau = %f" % (self._tau[x], upper_bound_tau)
            # slice sampling with inf stepping out limit
            slice_sample(self._tau[x], sample_func_tau, window_size=1.0, L_bound=0, R_bound=upper_bound_tau, step_out_limit=4)

            for y in range(self._A):
                # TODO: remove the assert
                upper_bound_nu = 1.0 / self._tau[x] - (sum(self._nu[:,x]) - self._nu[y,x])
                ## TODO: remove
                upper_bound_nu = min(upper_bound_nu, 10)
                assert self._nu[y,x] <= upper_bound_nu, "nu = %f, upper_bound_nu = %f" % (self._nu[y,x], upper_bound_nu)
                # slice sampling with inf stepping out limit
                slice_sample(self._nu[y,x], sample_func_nu, window_size=5.0, L_bound=0, R_bound=upper_bound_nu, step_out_limit=4)

        return self.get_tau(), self.get_nu()


    def sample_baseline_rate(self, use_prior=False):
        '''
        Update self.baseline_rate with one slice sampling draw

        Return the draw of self._baseline_rate
        '''
        def sample_func_baseline_rate(baseline_rate):
            self.set_baseline_rate(baseline_rate, x)
            if use_prior:
                return self.log_likelihood() + self.log_prior()
            else:
                return self.log_likelihood()

        for x in range(self._A):
            # slice sampling with inf stepping out limit
            slice_sample(self._baseline_rate[x], sample_func_baseline_rate, window_size=self._baseline_rate_window_size, L_bound=0, step_out_limit=4)

        return self.get_baseline_rate()

    def plot_rate_function(self, fig_title=None, show_figure=True, agents=None):
        '''
        Plot the rate function with the current stored parameters & 
        the integrated rate function.
        '''
        print ("\nplotting ...")
        fig = plt.figure()
        t_vec = linspace(0, self._T, num=1000)
        if agents is None: agents = range(self._A)
        a = len(agents)
        for j, x in enumerate(agents):
            ax = fig.add_subplot(a, 2, 2*j + 1)
            lambda_vec = array([self.rate(x, t) for t in t_vec])
            ax.plot(t_vec, lambda_vec)
            for event_time in self._data[x]:
                ax.axvline(x=event_time, ls="--", color="g")
            ax.set_xlim(0, self._T)
            ax.set_title("agent %d" % (x+1))

            ax2 = fig.add_subplot(a, 2, 2*j + 2)
            integrated_rate_vec =  array([self.integratedRate(x, t) for t in t_vec])
            event_count_vec = array([self.eventCount(x, t) for t in t_vec])
            ax2.plot(t_vec, event_count_vec, color="r", ls='--')
            ax2.plot(t_vec, integrated_rate_vec, color='b')
            ax2.set_xlim(0, self._T)
            ax2.set_title("%d events" % (len(self._data[x])))
        
        plt.tight_layout(pad=1, h_pad=1, w_pad=1)
        if fig_title is not None: fig.suptitle(fig_title)
        if show_figure: 
            plt.show()
            print("Plot shown")


    def read_hawkes_samples(self, hawkes_sample_dir):
        '''
        Read hawkes samples from a dir

        Return: nu_samples, tau_samples, baseline_rate_samples, ll_samples, ll_samples_test_set
        '''
        nu_samples = loadtxt(hawkes_sample_dir + "SAMPLE-nu.txt", skiprows=1)
        tau_samples = loadtxt(hawkes_sample_dir + "SAMPLE-tau.txt", skiprows=1)
        baseline_rate_samples = loadtxt(hawkes_sample_dir + "SAMPLE-baseline-rate.txt", skiprows=1)
        ll_samples = loadtxt(hawkes_sample_dir + "SAMPLE-ll.txt", skiprows=1)
        ll_samples_test_set = loadtxt(hawkes_sample_dir + "SAMPLE-ll-test-set.txt", skiprows=1)

        return nu_samples, tau_samples, baseline_rate_samples, ll_samples, ll_samples_test_set

    def test_goodness_of_fit(self):
        '''
        QQ plot of the residual process to assess the goodness of fit. 

        For a correct fit, the residual process should be a Poisson process of rate 1.
        '''
        residual_processes = []
        rate_mle = self.mle_homogenous_poisson()

        for x in range(self._A):
            this_process = array([self.integratedRate(x, t) for t in self._data[x]])
            residual_processes.append(this_process)

        # testing by exponential interval
        fig = plt.figure()
        for x in range(self._A):
            intervals = residual_processes[x][1:] - residual_processes[x][0:-1]
            ax = fig.add_subplot(2,self._A,x+1)
            probplot(intervals, dist="expon", plot=plt, fit=False)
            xvec = linspace(0,max(intervals)*1.1,num=100)
            ax.plot(xvec, xvec, color="g", ls="--")
            ax.set_title("residual intervals of agent %d" % (x+1))
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(0,xmax)
            ax.set_ylim(0,xmax)
            ax.grid()
            ## meanwhile testing gof of the MLE homogenous poisson
            residuals = rate_mle[x] * (array(self._data[x])[1:] - array(self._data[x])[0:-1])
            ax2 = fig.add_subplot(2,self._A,self._A+x+1)
            probplot(residuals, dist="expon", plot=plt, fit=False)
            xvec = linspace(0,max(residuals)*1.1,num=100)
            ax2.plot(xvec, xvec, color="g", ls="--")
            ax2.set_title("agent %d under homogenous (Poisson) fitting" % (x+1))
            xmin, xmax = ax2.get_xlim()
            ax2.set_xlim(0,xmax)
            ax2.set_ylim(0,xmax)            
            ax2.grid()
        
        plt.tight_layout()
        plt.show()

    def mle_homogenous_poisson(self):
        '''
        MLE of rate with homogenous poisson process
        '''
        rate_mle = zeros(self._A)
        for x in range(self._A):
            rate_mle[x] = len([t for t in self._data[x] if t <= self._time_to_split]) * 1.0 / self._time_to_split

        return rate_mle

    def posterior_predictive_test_set(self, alpha, beta):
        '''
        with Gamma(alpha, beta) prior on homogenous rate, 
        return the posterior predictive of the test set with a homogenous poisson process model.
        '''
        delta_t = self._T - self._time_to_split
        p = delta_t / (beta + self._T)

        pred = 0
        for x in range(self._A):
            N_T = len(self._data[x])
            N_train = len([t for t in self._data[x] if t <= self._time_to_split])
            N_test = N_T - N_train
            pred += -log(delta_t) + gammaln(N_T + alpha) - gammaln(alpha+N_train) + N_test * log(p) + (N_train + alpha) * log(1-p)
            pred -= N_test * log(delta_t)

        return pred

    def posterior_likelihood_train_set(self, alpha, beta):
        '''
        with Gamma(alpha, beta) prior on homogenous rate, 
        return the posterior likelihood of the training set with a homogenous poisson process model.
        '''
        p = 1.0/(beta/self._time_to_split + 2)

        pred = 0
        for x in range(self._A):
            N_train = len([t for t in self._data[x] if t <= self._time_to_split])
            pred += -log(self._time_to_split) + gammaln(2*N_train + alpha) - gammaln(N_train+alpha) + N_train * log(p) + (N_train+alpha) * log(1-p)
            pred -= N_train * log(self._time_to_split)

        return pred

    def MLE_predictive_test_set(self):
        '''
        Return the log likelihood of the test set under MLE estimates with a homogenous poisson process model
        '''
        rate_mle = self.mle_homogenous_poisson()
        delta_t = self._T - self._time_to_split

        pred = 0
        for x in range(self._A):
            N_T = len(self._data[x])
            N_train = len([t for t in self._data[x] if t <= self._time_to_split])
            N_test = N_T - N_train
            pred += N_test * log(rate_mle[x]) - rate_mle[x] * delta_t

        return pred

    def MLE_predictive_train_set(self):
        '''
        Return the log likelihood of the training set under MLE estimates with a homogenous poisson process model
        '''
        rate_mle = self.mle_homogenous_poisson()

        ll = 0
        for x in range(self._A):
            N_train = len([t for t in self._data[x] if t <= self._time_to_split])
            ll += N_train * log(rate_mle[x]) - rate_mle[x] * self._time_to_split

        return ll


# class over

def unitest_simulation():
    A = 2
    T = 200.0
    print ("A = ", A)
    print ("max time T = ", T)
    hawkes_proc = Hawkes(A=A, T=T)

    # parameters
    nu = 1.0 / array([[1.5, 8], [4, 2]])
    time_decay = 0.8/ array([0.8, 1])
    baseline_rate = array([0.4, 0.6])
    print ("stationarity:", hawkes_proc.test_stationarity())
    print ("strong stationarity (matrix norm constraint): ", hawkes_proc.test_matrix_norm_constraint())

    # simulate
    hawkes_proc.set_nu(nu)
    hawkes_proc.set_tau(time_decay)
    hawkes_proc.set_baseline_rate(baseline_rate)
    data = hawkes_proc.simulate()
    hawkes_proc.loadData(data)
    hawkes_proc.plot_data_events()
    hawkes_proc.plot_rate_function()
    print ("\n* True parameters")
    print ("ll_train =", hawkes_proc.log_likelihood(), "ll_test =", hawkes_proc.log_likelihood_test_set())
    hawkes_proc.test_goodness_of_fit()

    # MLE estimate
    hawkes_proc.randomize_parameters()
    MLE_baseline_rate, MLE_tau, MLE_nu = hawkes_proc.optimize_parameters()
    print ("\n* MLE")
    print ("baseline rate: ", "True =", baseline_rate, "\nMLE =", MLE_baseline_rate)
    print ("time decay: ", "True =", time_decay, "\nMLE =", MLE_tau)
    print ("influence matrix: ", "True =", nu, "\nMLE =", MLE_nu)
    print ("ll_train =", hawkes_proc.log_likelihood(), "ll_test =", hawkes_proc.log_likelihood_test_set()    )
    hawkes_proc.test_goodness_of_fit()

    # MCMC
    B = 100
    N = 100
    ## containers
    nu_samples = zeros((N, A, A))
    tau_samples = zeros((N, A))
    baseline_rate_samples = zeros((N, A))
    ll_samples = zeros(N)
    ll_test_samples = zeros(N)
    ## sample
    for k in range(B+N):
        if k%20==0: print ("Sampling %d out of %d ..." % (k, B+N))
        if k>=B:
            nu_samples[k-B, ...] = hawkes_proc.sample_nu()
            tau_samples[k-B, ...] = hawkes_proc.sample_tau()
            baseline_rate_samples[k-B, ...] = hawkes_proc.sample_baseline_rate()
            ll_samples[k-B] = hawkes_proc.log_likelihood()
            ll_test_samples[k-B] = hawkes_proc.log_likelihood_test_set()

    ## plot
    print ("\n* MCMC")
    print ("ll_train =", mean(ll_samples), "ll_test =", mean(ll_test_samples))
    print ("influence matrix")
    print ("True:", nu)
    print ("sample mean:", mean(nu_samples, axis=0))
    print ("")
    print ("time decay")
    print ("True:", time_decay)
    print ("sample mean:", mean(tau_samples, axis=0))
    print ("")
    print ("baseline rate")
    print ("True:", baseline_rate)
    print ("sample mean: ", mean(baseline_rate_samples, axis=0))

    hawkes_proc.set_nu(mean(nu_samples, axis=0))
    hawkes_proc.set_tau(mean(tau_samples, axis=0))
    hawkes_proc.set_baseline_rate(mean(baseline_rate_samples, axis=0))

    hawkes_proc.test_goodness_of_fit()



if __name__ == '__main__':
    unitest_simulation()

