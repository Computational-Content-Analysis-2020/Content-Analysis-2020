# -*- coding: utf-8 -*-
"""
Created on Oct 17 2014

@author: richard
"""

import sys
sys.path.insert(0, "/home/home3/guo/.local/lib/python2.7/site-packages") # dirty hack to use local numpy/scipy packages of latest version

from numpy import zeros, identity, bincount, log, exp, abs, sqrt, savez, savetxt, shape, array, mean, median, hstack, vstack, ones
from numpy.random import random, gamma, dirichlet, multinomial
from scipy.special import psi as digamma

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from bec import BEC, Message

def sample_bec(bec, display=20, sample_number=500, burn_in_number=500, resultDirName="./"):
    A = bec._A
    V = bec._V

    # containers
    log_prior_and_log_likelihood = zeros((sample_number, 3))
    time_decays = zeros((sample_number,A))
    influences = zeros((sample_number,A,A))
    word_concentrations = zeros((sample_number, A))
    word_pseudocounts = zeros((sample_number, A, V))

    # sample
    for i in range(0, sample_number+burn_in_number):
        if (display >0 and i % display==0):
            print ("sampling %d out of %d" % (i+1, sample_number+burn_in_number))

        if display==1: print ("Sampling time decay...")
        time_decay = bec.sample_time_decay()
        if display==1: print ("Sampling influence...")
        influence = bec.sample_influence()
        if display==1: print ("Sampling concentration...")
        concentration = bec.sample_word_concentration()
        if display==1: print ("Sampling pseudocounts...")
        for l in range(10):
            pseudocounts = bec.sample_word_pseudocounts_multivariate()

        if i>=burn_in_number:
            time_decays[i-burn_in_number, :] = time_decay
            influences[i-burn_in_number, :, :] = influence
            word_concentrations[i-burn_in_number, :] = concentration
            word_pseudocounts[i-burn_in_number, :, :] = pseudocounts
            log_prior_and_log_likelihood[i-burn_in_number, :] = [bec.log_prior(), bec.log_likelihood(), bec.log_likelihood_test_set()]

    # save to file

    outputfilename_influence = resultDirName + "SAMPLE-influence.txt"
    outputfilename_time_decay = resultDirName + "SAMPLE-time_decay.txt"
    outputfilename_pseudocounts = resultDirName + "SAMPLE-pseudocounts.txt"
    outputfilename_word_concentration = resultDirName + "SAMPLE-word_concentration.txt"
    outputfilename_logprior_and_loglikelihood = resultDirName + "SAMPLE-log_prior_and_log_likelihood.txt"

    # log prob
    savetxt(outputfilename_logprior_and_loglikelihood, log_prior_and_log_likelihood, header="log.prior\tlog.likelihood\tlog.likelihood.test.set", comments='')
    # time decay
    timeDecayHeader = "".join(["time.decay.%d\t" %(i) for i in range(1, A+1)])
    savetxt(outputfilename_time_decay, time_decays, header=timeDecayHeader, comments='')
    # word concentrations
    wordCocentrationHeader = "".join(["word.concentration.%d\t" %(i) for i in range(1, A+1)])
    savetxt(outputfilename_word_concentration, word_concentrations, comments='', header=wordCocentrationHeader)
    # influences
    influenceHeaders = "%s\t" * (A*A) % tuple(["influence.%d.%d\t" % (i,j) for i in range(1,A+1) for j in range(1,A+1)])
    savetxt(outputfilename_influence, 
            influences.reshape((shape(influences)[0], A*A)), header=influenceHeaders, comments='')
    # word pseudocounts
    wordPseudocountsHeader = "".join(["word.pseudocount.%d.for.word.%d\t" %(i,j) for i in range(1, A+1) for j in range(1,V+1)])
    savetxt(outputfilename_pseudocounts, word_pseudocounts.reshape((shape(word_pseudocounts)[0], A*V)), 
        comments='', header=wordPseudocountsHeader)
    
    print ("\n%d mcmc samples saved to file." % (sample_number))   

    # return the samples
    return (log_prior_and_log_likelihood, 
        time_decays, 
        influences, 
        word_concentrations, 
        word_pseudocounts)
