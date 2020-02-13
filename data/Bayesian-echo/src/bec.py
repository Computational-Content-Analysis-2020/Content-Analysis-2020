# configurable bec: 
# configure diagonal enabled/disabled
# configure prior used or not

from numpy import zeros, identity, bincount, log, exp, abs, sqrt, savez, savetxt, shape, eye, zeros, ones, all, any, diag
from numpy.random import random, gamma, dirichlet, multinomial, poisson
from scipy.special import psi as digamma
import os
import os.path
from likelihoods import log_gamma, log_polya
from slice_sampler import slice_sample, multivariate_slice_sample


class BEC(object):

    def __init__(self, use_prior=False, non_diagonal=False, time_decay_prior=None, influence_prior=None, word_concentration_prior=None, word_pseudocount_prior=None):
        ''' Set the priors: all of them are parmeters for Gamma priors of the form (shape a, scale b)
        '''
        if use_prior:
            assert time_decay_prior is not None
            assert influence_prior is not None
            assert word_concentration_prior is not None
            assert word_pseudocount_prior is not None
            self._time_decay_prior = time_decay_prior
            self._influence_prior = influence_prior
            self._word_concentration_prior = word_concentration_prior
            self._word_pseudocount_prior = word_pseudocount_prior

        self._use_prior = use_prior
        self._non_diagonal = non_diagonal

        # set bounds
        self._time_decay_upper_bound = 500
        self._influence_upper_bound = 10
        self._word_concentration_upper_bound = 10
        self._word_pseudocount_upper_bound = 500

        print ("\nBEC intialized (use_prior=%s, non_diagonal=%s)\n" % (use_prior, non_diagonal))

    def set_time_decay_upper_bound(self, time_decay_upper_bound):
        self._time_decay_upper_bound = time_decay_upper_bound

    def set_influence_upper_bound(self, influence_upper_bound):
        self._influence_upper_bound = influence_upper_bound

    def set_word_concentration_upper_bound(self, word_concentration_upper_bound):
        self._word_concentration_upper_bound = word_concentration_upper_bound

    def set_word_pseudocount_upper_bound(self, word_pseudocount_upper_bound):
        self._word_pseudocount_upper_bound = word_pseudocount_upper_bound

    def init_parameters(self, A, V):
        '''
        Initialize parameters by setting A, V and sampling other parameters from priors:
        self._time_decay, self._influence, self._word_concentration, self._word_pseudocounts

        This works ONLY when no message has been stored. Use self.randomize_parameters() if with messages.
        '''

        self._A = A 
        self._V = V
        
        if self._use_prior:
            # agent-specific time decay
            self._time_decay = gamma(*self._time_decay_prior, size=(A))
            # A x A influence matrix, iid drawn from Gamma(_influence_prior)
            self._influence = gamma(*self._influence_prior, size=(A, A))
            # one scalar for each agent, iid drawn from Gamma(_word_concentration_prior)
            self._word_concentration = gamma(*self._word_concentration_prior, size=A)
            # A x V, with each row being the agents' word-freq profile
            # each row is a vector of length V, each element being a pseudocount iid drawn from Gamma(_word_pseudocount_prior)
            self._word_pseudocounts = gamma(*self._word_pseudocount_prior, size=(A,V))
        else:
            self._time_decay = 5.0 * ones(A)
            self._influence = 2.0 * ones((A,A))
            self._word_concentration = 100.0 * ones(A)
            self._word_pseudocounts = 1.0 * ones((A, V))

        # diagonals or not?
        if self._non_diagonal:
            for x in range(self._A):
                self._influence[x,x] = 0

        

    def generate(self, total_number_of_utterances=100, average_message_length=30, time_to_split=None):
        '''
        Generate messages.

        '''
        A = self._A
        V = self._V

        message_lengths = []
        for i in range(total_number_of_utterances):
            message_lengths.append(poisson(average_message_length))


        T = 100.0
        delta_t = T /(sum(message_lengths)+3*total_number_of_utterances)

        self._messages=[[] for x in range(A)]

        sender = -1
        t = 0.0
        sender_names = ["sender%s" % (x) for x in range(A)]
        for i in range(total_number_of_utterances):
            sender = (sender+1) % A
            sender_name = sender_names[sender]
            start_time = t
            end_time = start_time + message_lengths[i] * delta_t
            t = end_time + 3 * delta_t
            self._messages[sender].append(Message(sender=sender, sender_name=sender_name, 
                start_time=start_time, end_time=end_time, A=A, V=V, T=T, token_type_counts=None))
            self._messages[sender][-1]._total_token_type_counts = message_lengths[i]

        # internal Data
        # self._messages[x] is the list of Messages sent from x
        
        # excitation_pseudocounts[i,j,v] = sum of weighted counts of token v from past i -> j messages
        # the weights are based on exponential time decay kernel
        excitation_pseudocounts = zeros((A,A,V))
        last_message_time = 0.0
        
        # generator, first outside loop, then inside loop
        messages = (message for message_list in self._messages for message in message_list)
        # messages generated one by one in time order
        for message in sorted(messages, key=lambda message: message.get_start_time()):
            # x is the sender
            x = message.get_sender()
            t = message.get_start_time()
            
            # efficient update of excitation pseudocounts 
            # (i) decay of ALL old pseudocounts by multiplyting the time decay factors from last t to current t
            for y in range(self._A):
                excitation_pseudocounts[:,y,:] *= exp(-(t - last_message_time)/self._time_decay[y])
            
            last_message_time = message.get_end_time()

            # influence on x from all agents 
            message.set_influence(self._influence[:,x])
            # word concentration scalar for x
            message.set_concentration(self._word_concentration[x])
            # message generated 
            # based on (1) sender x's language profile, i.e. its word_pseudocounts, vector of length V
            message.set_pseudocounts(self._word_pseudocounts[x,:])
            # (2) excitation pseudocounts that x receives from all agents, up to now
            # characterized by a (A x V) matrix
            message.set_excitation_pseudocounts(excitation_pseudocounts[:,x,:], senders=None)
            
            message.generate(message._total_token_type_counts)
            
            # print "sender", x
            # raw_input("~")
            # print "before"
            # print excitation_pseudocounts
            # (ii) add new pseudocounts (t=t) without decay
            # added for entries for all agents received from agent x 
            # because this message is broadcasted to everyone including x itself
            # print "after"
            excitation_pseudocounts[x,:,:] += message.count_token_types()
            # print excitation_pseudocounts
            # print message.count_token_types()

        if time_to_split is None:
            self._time_to_split = last_message_time * 0.9

        self.__ll_cached = False
    
    def import_data(self, V, A, messages, time_to_split=100.0):
        '''
        Import data into the bec object. 

        messages: a list of A lists of messages, with each list contains all the messages sent from one agent.

        time_to_split: all messages with t <= time_to_split go to the training set, with the rest go to the test set. 
        The default time_to_split is 100.0

        '''
        self._A = A 
        self._V = V

        assert A==len(messages)
        
        self._messages = messages[:]
        self._time_to_split = time_to_split

        self._training_message_number = 0
        self._test_message_number = 0
        for messageList in self._messages:
            for message in messageList:
                start_time = message.get_start_time()
                end_time = message.get_end_time()
                if end_time<=self._time_to_split:
                    self._training_message_number += 1
                elif start_time >= self._time_to_split:
                    self._test_message_number += 1
        self._total_message_number = self._training_message_number + self._test_message_number

        print ("\nLoaded %d messages in total" % (self._total_message_number))
        print ("Training set (before t=%.2f): %d messages (%.2f%%)" % (self._time_to_split, self._training_message_number, self._training_message_number*100.0/self._total_message_number))
        print ("Test set (after t=%.2f) %d messages (%.2f%%)" % (self._time_to_split, self._test_message_number, self._test_message_number*100.0/self._total_message_number))
        
        self.randomize_parameters()
        
        self.__iteration = 0
        self.__ll_cached = False
    
    def update_excitation_pseudocounts(self, senders, recipients):
        '''
        Data update: excitation_pseudocounts. 
        Recompute each message's excitation_pseudocounts in self._messages, by iterating over all messages

        senders: None, if update all senders

        recipients: None, if update all recipients

        UPDATE: compute decay with (t_start - t_end) instead of just t
        '''
        senders = (range(self._A) if senders is None else [senders])
        recipients = (range(self._A) if recipients is None else [recipients])
        
        excitation_pseudocounts = zeros((self._A, self._A, self._V))
        last_message_time = 0
        
        # messages from self._messages if it is sent from/to agent in senders or recipients        
        messages = (m for user, message_list in enumerate(self._messages) if user in senders or user in recipients for m in message_list)
        for message in sorted(messages, key=lambda message: message.get_start_time()):
            x = message.get_sender()
            t_start = message.get_start_time()
            t_end = message.get_end_time()
            
            # consider only decay from start time to start time
            # because the excitation pseudocount at start time of a msg is what is used to compute the token probabilities
            # for that msg
            for y in range(self._A):
                excitation_pseudocounts[:,y,:] *= exp(-(t_start - last_message_time)/self._time_decay[y])
            last_message_time = t_start
            
            # also update the excitation pseudocounts associated with that message
            # update only when the receiver is involved
            if x in recipients:
                message.set_excitation_pseudocounts(excitation_pseudocounts[senders,x,:], senders=senders)
            
            # because the excitation should be initiated at the end time of that msg
            # we can instead set the excited pseudocounts at start time to be an inflated version 
            # = token type counts / decay from t_start to t_end
            for y in range(self._A):
                excitation_pseudocounts[x,y,:] += message.count_token_types() * exp((t_end - t_start)/self._time_decay[y])
            
        self.__ll_cached = False
        
    def set_time_decay(self, time_decay, x):
        '''
        Data update: set self._time_decay[x] = time_decay
        it will call self.update_excitation_pseudocounts
        '''
        assert all(time_decay <= self._time_decay_upper_bound) and all(time_decay>0)

        if (self._time_decay[x,None] != time_decay).any():
            self._time_decay[x] = time_decay
            self.update_excitation_pseudocounts(senders=None, recipients=x)
            self.__ll_cached = False
        
    def set_influence(self, influence, x, y):
        '''
        Data update: set self._influence[x,y] = influence, 
        entails updating the message's influences in self._messages 
        if the sender of that message is included in y

        '''
        assert all(influence>=0) and all(influence <= self._influence_upper_bound)
        # diagonal
        if self._non_diagonal:
            if x is not None and y is not None: 
                assert x!=y
            elif x is None and y is None:
                assert all(diag(influence)==0)

        if (self._influence[x,y,None] != influence).any():
            self._influence[x,y] = influence
        
            for recipient in (range(self._A) if y is None else [y]):
                for message in self._messages[recipient]:
                    message.set_influence(self._influence[:,recipient])
                    
            self.__ll_cached = False
        
    def set_word_concentration(self, word_concentration, x):
        '''
        Data update: set self._word_concentration[x] = word_concentration, 
        entails updating the associated word_concentration in self._messages[x]
        '''
        assert all(word_concentration>0) and all(word_concentration <= self._word_concentration_upper_bound)

        if (self._word_concentration[x,None] != word_concentration).any():
            self._word_concentration[x] = word_concentration
        
            for sender in (range(self._A) if x is None else [x]):
                for message in self._messages[sender]:
                    message.set_concentration(self._word_concentration[sender])
                
            self.__ll_cached = False
    
    def set_word_pseudocounts(self, word_pseudocount, x, v):
        '''
        Data update: set self._word_pseudocounts[x,v] = word_pseudocount, 

        entails updating the associated word_pseudocounts in self._messages[x]
        '''
        assert all(word_pseudocount>=0) and all(word_pseudocount <= self._word_pseudocount_upper_bound), "%s" % (word_pseudocount)

        if (self._word_pseudocounts[x,v,None] != word_pseudocount).any():
            self._word_pseudocounts[x,v] = word_pseudocount
        
            for sender in (range(self._A) if x is None else [x]):
                for message in self._messages[sender]:
                    message.set_pseudocounts(self._word_pseudocounts[sender,:])

        self.__ll_cached = False
                
    def randomize_parameters(self):
        """
        Randomize the internal parameters and those associated with self._messages, 
        by drawing from priors

        Update: self._time_decay, self._influence, self._word_concentration, self._word_pseudocounts

        TODO: Get rid of this code duplication w/ self.generate!
        """
        A = self._A
        V = self._V

        if self._use_prior:
            while True:
                # agent-specific time decay
                self._time_decay = gamma(*self._time_decay_prior, size=(A))
                # A x A influence matrix, iid drawn from Gamma(_influence_prior)
                self._influence = gamma(*self._influence_prior, size=(A, A))
                # one scalar for each agent, iid drawn from Gamma(_word_concentration_prior)
                self._word_concentration = gamma(*self._word_concentration_prior, size=A)
                # A x V, with each row being the agents' word-freq profile
                # each row is a vector of length V, each element being a pseudocount iid drawn from Gamma(_word_pseudocount_prior)
                self._word_pseudocounts = gamma(*self._word_pseudocount_prior, size=(A,V))
                if all(self._word_pseudocounts <= self._word_pseudocount_upper_bound) and all(self._influence<=self._influence_upper_bound) and all(self._time_decay<=self._time_decay_upper_bound) and all(self._word_concentration<=self._word_concentration_upper_bound):
                    break
        else:
            self._time_decay = 5.0 * ones(A)
            self._influence = 2.0 * ones((A,A))
            self._word_concentration = 100.0 * ones(A)
            self._word_pseudocounts = 1.0 * ones((A, V))

        # diagonals or not?
        if self._non_diagonal:
            for x in range(self._A):
                self._influence[x,x] = 0
        
        # important: update associated parameters in each message
        for x, message_list in enumerate(self._messages):
            for message in message_list:
                message.set_influence(self._influence[:,x])
                message.set_concentration(self._word_concentration[x])
                message.set_pseudocounts(self._word_pseudocounts[x,:])
        
        self.update_excitation_pseudocounts(senders=None, recipients=None)
        
        self.__ll_cached = False

        print ("BEC parameters randomized")

    
    def sample(self, iterations, display=20):
        for s in range(iterations):
            if display==1: print ("sampling time decay...")
            self.sample_time_decay()
            if display==1: print ("sampling influence matrix...")
            self.sample_influence()
            if display==1: print ("sampling word concentration parameters...")
            self.sample_word_concentration()
            if display==1: print ("sampling word pseudocounts ...")
            self.sample_word_pseudocounts_multivariate()
                        
    
    def sample_time_decay(self, window_size=100, step_out_limit=0):
        def sample_func(time_decay):
            self.set_time_decay(time_decay, x)
            return self.log_prob()
        
        for x in range(self._A):
            slice_sample(self._time_decay[x], sample_func, window_size=window_size, L_bound=0, R_bound=self._time_decay_upper_bound, step_out_limit=step_out_limit)

        return self._time_decay.copy()
    
    def sample_influence(self, window_size=10, step_out_limit=2):
        def sample_func(influence):
            self.set_influence(influence, x, y)
            return self.log_prob()
            
        for x in range(self._A):
            for y in range(self._A):
                if self._non_diagonal:
                    if x==y:
                        continue
                slice_sample(self._influence[x,y], sample_func, window_size=window_size, L_bound=0, R_bound=self._influence_upper_bound, step_out_limit=step_out_limit)

        return self._influence.copy()
    
    def sample_word_concentration(self, window_size=200, step_out_limit=0):
        def sample_func(word_concentration):
            self.set_word_concentration(word_concentration, x)
            return self.log_prob()
        
        for x in range(self._A):
            slice_sample(self._word_concentration[x], sample_func, window_size=window_size, L_bound=1e-4, R_bound=self._word_concentration_upper_bound, step_out_limit=step_out_limit)

        return self._word_concentration.copy()
    
    def sample_word_pseudocounts(self, window_size=10, step_out_limit=0):
        def sample_func(word_pseudocount):
            self.set_word_pseudocounts(word_pseudocount, x, v)
            return self.log_prob()
        
        for x in range(self._A):
            for v in range(self._V):
                slice_sample(self._word_pseudocounts[x,v], sample_func, window_size=window_size, L_bound=0, R_bound=self._word_pseudocount_upper_bound, step_out_limit=step_out_limit)

        return self._word_pseudocounts.copy()

    def sample_word_pseudocounts_multivariate(self, window_size=500):
        def sample_func(word_pseudocounts):
            self.set_word_pseudocounts(word_pseudocounts, x, v=None)
            return self.log_prob()

        for x in range(self._A):
            multivariate_slice_sample(self._word_pseudocounts[x,:], sample_func, window_size=window_size, L_bound=1e-4, R_bound=self._word_pseudocount_upper_bound, gradient_func=None)

        return self._word_pseudocounts.copy()

    def save_state(self, directory):
        params = {
                  'time_decay': self._time_decay,
                  'influence': self._influence,
                  'word_concentration': self._word_concentration,
                  'word_pseudocounts': self._word_pseudocounts,
                  'log_prob': self.log_prob()
        }
        
        savez(os.path.join(directory, '%i.npz' % self.__iteration), **params)

    def log_likelihood(self):
        if self.__ll_cached:
            return self.__log_likelihood
        else:
            self.log_prob()     # update cache
            return self.__log_likelihood

    def log_prior(self):
        if self.__ll_cached:
            return self.__log_prior
        else:
            self.log_prob()     # update cache
            return self.__log_prior

    def log_likelihood_test_set(self):
        if self.__ll_cached:
            return self.__log_likelihood_test_set
        else:
            self.log_prob()     # update cache
            return self.__log_likelihood_test_set
    
    def log_prob(self):
        '''
        Return the log_prob of the Training Set = log_likelihood + log_prior

        Running log_prob will cache the log_likelihood, log_prior (for training set) and log_likelihood_test_set if they are not cached
        '''
        if not self.__ll_cached:
            self.__ll = 0

            # intermediate variables storing log_likelihood and log_prior seperately
            # ll of the training set
            self.__log_likelihood = sum(message.log_prob() for message_list in self._messages for message in message_list if message.get_end_time()<=self._time_to_split)
            # ll of the test set
            self.__log_likelihood_test_set = sum(message.log_prob() for message_list in self._messages for message in message_list if message.get_start_time()>self._time_to_split)
            # 
            if self._use_prior:
                self.__log_prior = 0
                self.__log_prior += log_gamma(self._time_decay, *self._time_decay_prior).sum()
                if self._non_diagonal:
                    # off-diagonal entries
                    self.__log_prior += log_gamma(self._influence[eye(self._A)==0], *self._influence_prior).sum()
                else:
                    self.__log_prior += log_gamma(self._influence, *self._influence_prior).sum()
                self.__log_prior += log_gamma(self._word_concentration, *self._word_concentration_prior).sum()
                self.__log_prior += log_gamma(self._word_pseudocounts, *self._word_pseudocount_prior).sum()
            else:
                # no prior
                self.__log_prior = 0
            
            self.__ll = self.__log_likelihood + self.__log_prior            

            self.__ll_cached = True
        
        return self.__ll


    def log_prob_pseudocount_grad(self):
        shape, scale = map(float, self._word_pseudocount_prior)
        
        grad = (shape - 1) / self._word_pseudocounts - 1 / scale
        
        for x in range(self._A):
            grad[x,:] += sum(message.log_prob_pseudocount_grad() for message in self._messages[x])
        
        return grad

# revise the Message object to distinguish between start and end time of each utterance
# revised Oct 4, 2014


class Message(object):

    def __init__(self, sender, sender_name, start_time, end_time, A, V, T, tokens=None, token_type_counts=None):
        assert start_time < end_time and end_time <= T, "Error times: start_time=%s, end_time=%s, T=%s" % (start_time, end_time, T)
        self._sender = sender
        self._sender_name = sender_name
        self._start_time = start_time
        self._end_time = end_time
        self._V = V
        self._A = A
        self._T = T
        self._pseudocounts = None
        self._excitation_pseudocounts = zeros((A, V))
        
        if tokens is not None:
            self._tokens = tokens
            self._token_type_counts = bincount(self._tokens, minlength=self._V)
        
        if token_type_counts is not None:
            assert tokens is None
            self._token_type_counts = token_type_counts
            self._total_token_type_counts = sum(self._token_type_counts)
        
        self.__ll_cached = False
        self.__base_measure_cached = False
    
    def generate(self, doc_length):
        self._token_prob = dirichlet(self._concentration * self.base_measure())
        self._tokens = multinomial(1, self._token_prob, size=doc_length).argmax(axis=-1)
        self._token_type_counts = bincount(self._tokens, minlength=self._V)
        self.__ll_cached = False
        
    def get_sender(self):
        return self._sender

    def get_sender_name(self):
        return self._sender_name
        
    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        return self._end_time

    def get_V(self):
        return self._V

    def get_A(self):
        return self._A

    def get_T(self):
        return self._T

    def count_token_types(self):
        return self._token_type_counts

    def get_total_token_counts(self):
        return self._total_token_type_counts
    
    def get_weighted_excitation_pseudocounts(self):
        '''
        Return the weighted excitation pseudocounts in shape A x V, where the u-th row 
        is the excitation_pseudocounts from u multiplied by the influence coeff from u. 

        Important: balance the magnitudes of weighted_excitation_pseudocounts and word_pseudocounts
        '''
        weighted_excitation_pseudocounts = zeros((self._A, self._V))
        for u in range(0, self._A):
            weighted_excitation_pseudocounts[u,:] = self._influence[u] * self._excitation_pseudocounts[u,:]
        return weighted_excitation_pseudocounts

    def base_measure(self):
        if not self.__base_measure_cached:
            
            self.__base_measure = self._pseudocounts.copy() + self._influence.dot(self._excitation_pseudocounts)
            self.__base_measure /= self.__base_measure.sum()
            self.__base_measure_cached = True
        
        return self.__base_measure

    def set_concentration(self, concentration):
        self._concentration = concentration
        self.__ll_cached = False
    
    def set_influence(self, influence):
        self._influence = influence
        self.__base_measure_cached = False
        self.__ll_cached = False
        
    def set_pseudocounts(self, pseudocounts):
        self._pseudocounts = pseudocounts
        self.__base_measure_cached = False
        self.__ll_cached = False
        
    def set_excitation_pseudocounts(self, pseudocounts, senders):
        self._excitation_pseudocounts[senders,:] = pseudocounts
        self.__base_measure_cached = False
        self.__ll_cached = False
    
    def log_prob(self):
        if not self.__ll_cached:
            self.__ll = log_polya(self._token_type_counts, self._concentration * self.base_measure())
            self.__ll_cached = True
        
        return self.__ll

    def log_prob_pseudocount_grad(self):
        unnormalized_base_measure = self._pseudocounts.copy() + self._influence.dot(self._excitation_pseudocounts)
        unnormalized_base_measure_sum = unnormalized_base_measure.sum()
        dirichlet_param = self._concentration * unnormalized_base_measure / unnormalized_base_measure_sum
        
        deriv = identity(self._V) * (unnormalized_base_measure_sum - unnormalized_base_measure) + (1 - identity(self._V)) * (-unnormalized_base_measure) 
        deriv /= unnormalized_base_measure_sum ** 2
        
        grad = self._concentration * (deriv * (digamma(self._token_type_counts + dirichlet_param) - digamma(dirichlet_param))).sum(axis=1)
        
        return grad

    def __repr__(self):
        me = {}
        me["sender"] = self._sender
        me["sender_name"] = self._sender_name
        me["total_token_type_counts"] = self._total_token_type_counts
        me["start_time"] = self._start_time
        me["end_time"] = self._end_time
        me["A"] = self._A
        me["V"] = self._V
        me["ll_cached"] = self.__ll_cached
        me["ll"] = self.__ll
        me["token_type_counts"] = self._token_type_counts
        me["concentration"] = self._concentration
        me["pseudocounts"] = self._pseudocounts
        me["excitation pseudocounts"] = self._excitation_pseudocounts

        return me

    def __str__(self):
        s = ""
        s += "\nSender: %s" % (self.get_sender())
        s += "\nSender name: %s" % (self.get_sender_name())
        s += "\nStart time: %s" % (self.get_start_time())
        s += "\nEnd time: %s" % (self.get_end_time())
        s += "\nTotal token counts: %s" % (self._total_token_type_counts)        
        s += "\nToken counts: %s" % (self._token_type_counts)
        s += "\nToken proportions: %s" % (self._token_type_counts * 1.0 / self._token_type_counts.sum())
        s += "\npseudocounts:%s" % (self._pseudocounts)
        s += "\nExcitation pseudocounts: %s" % (self._excitation_pseudocounts)
        
        return s

def main():

    A=3
    V=3
    doc_length=100
    
    T = 100
    num_messages = 100
    message_times = random((A,num_messages))
    message_times *= T
    message_times.sort()
    
    
    samples = 1000
    
    # set priors
    bec = BEC(use_prior=False, non_diagonal=False)
    bec.init_parameters(A, V)
    # generate broadcast messages
    bec.generate(message_times=message_times, doc_length=doc_length)

    print ("BEC synthetic data generated")
    bec.sample(100, display=1)
    
    
if __name__ == '__main__':
    main()
    