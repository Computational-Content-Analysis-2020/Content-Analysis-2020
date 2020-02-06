# -*- coding: utf-8 -*-
"""
Created on Sept 30 2014

@author: richard
"""
from numpy import zeros, identity, bincount, log, exp, abs, sqrt, savetxt, shape, array, mean, median, hstack, vstack, ones

from bec import BEC
from bec_sampler import sample_bec

from talkbankXMLparse import talkbankXML

import os
import sys

def main():

    # priors
    time_decay_prior = (100, 1)
    influence_prior = (0.1, 10)
    word_concentration_prior = (10, 1)
    word_pseudocount_prior = (10, 1)

    # constants
    V = int(sys.argv[2])#600
    T = 100.0
    remove_stop_words = True
    language = sys.argv[3]

    use_prior = True
    non_diagonal = True
    # a speaker uttering fewer than this number will be removed
    min_number_msg = 5
    merge_consecutive_from_same_sender = True

    # samples and burn-in's
    burn_in_number = 500
    sample_number = int(sys.argv[4]) # number of sampling times
    display = 1  # rounds for one display, display=1 for verbose display

    print ("\n << BEC >>")


    # initialize bec object with prior parameters
    bec = BEC(use_prior=use_prior, non_diagonal=non_diagonal,
              time_decay_prior=time_decay_prior,
              influence_prior=influence_prior,
              word_concentration_prior=word_concentration_prior,
              word_pseudocount_prior=word_pseudocount_prior)

    # load messages from data

    #dataDir = "../data/12-angry-men/"

    dataFileName_short = sys.argv[1]#"test.xml"  #"12-angry-men.xml"

    #dataFileName = dataDir + dataFileName_short
    dataFileName = '.'+os.path.sep+'data'+os.path.sep+dataFileName_short.split('.')[0]+os.path.sep+dataFileName_short

    print ("Loading XML data from ", dataFileName, "...")
    print ("Working with V =", V, "T =", T)
    talkbankxml_instance = talkbankXML(dataFileName,
                                       remove_stop_words=remove_stop_words,
                                       min_number_msg=min_number_msg,
                                       merge_consecutive_from_same_sender=merge_consecutive_from_same_sender,
                                       language = language)
    messages_all = talkbankxml_instance.exportMessages(
        V=V, maxTime=T, display=False)
    A = len(messages_all)

    # import data into bec
    # determine the time to split the training and test set
    test_set_percentage = 0.1
    allMessages = [msg for messageList in messages_all for msg in messageList]
    allMessages = sorted(allMessages, key=lambda msg: msg.get_start_time())
    time_to_split = allMessages[
        int((1 - test_set_percentage) * len(allMessages)) + 1].get_start_time()

    bec.init_parameters(A=A, V=V)
    bec.import_data(
        V=V, A=A, messages=messages_all, time_to_split=time_to_split)

    # all messages in bec's training set/test set
    allMessages_train_set = (
        msg for messageList in bec._messages for msg in messageList if msg.get_end_time() <= bec._time_to_split)
    allMessages_train_set = sorted(
        allMessages_train_set, key=lambda msg: msg.get_start_time())
    allMessages_test_set = (
        msg for messageList in bec._messages for msg in messageList if msg.get_start_time() > bec._time_to_split)
    allMessages_test_set = sorted(
        allMessages_test_set, key=lambda msg: msg.get_start_time())

    print ("Data imported into bec: V = %d, A = %d, T=%.2f, split time=%.5f" % (bec._V, bec._A, T, time_to_split))
    print ("Training set: %d messages, test set: %d messages" % (len(allMessages_train_set), len(allMessages_test_set)))

    # path to save results
    if os.path.sep in dataFileName_short:
        result_path = dataFileName_short.split(os.path.sep)[-1].split(".")[0]
    else:
        result_path = dataFileName_short.split(".")[0]
    resultDirName = "./results/" + result_path + "/"

    if not os.path.exists(resultDirName):
        os.makedirs(resultDirName)

    print ("Working under", resultDirName)

    # save a list of agent names
    talkbankxml_instance.save_cast_table(resultDirName)

    # empirical trends of the data
    # data
    sender_and_time_array = zeros((len(allMessages), 2))
    token_proportions_array = zeros((len(allMessages), V))

    for i, msg in enumerate(allMessages):
        # print "MSG %d" % (i+1)
        # msg.print_message()
        sender_and_time_array[i, 0] = msg.get_sender() + 1  # start from 1
        sender_and_time_array[i, 1] = msg.get_start_time()
        token_proportions_array[
            i, :] = msg._token_type_counts * 1.0 / msg._token_type_counts.sum()

    # write to file
    v_header = "".join(["word.%d\t" % (k) for k in range(1, V + 1)])

    # save empirical trends of real data
    savetxt(resultDirName + "sender_and_time.txt", sender_and_time_array,
            header="sender\ttime", fmt="%d %.2f", comments='')
    savetxt(resultDirName + "token_proportions.txt",
            token_proportions_array, header=v_header, fmt="%.3f", comments='')

    # save meta information
    fw = open(resultDirName + "meta-info.txt", "w")
    print ("Data:", dataFileName, file=fw)
    print ("Number of mcmc samples:", sample_number, file=fw)
    print ("Number of mcmc burn-ins:", burn_in_number, file=fw)
    print ("V:", V, file=fw)
    print ("A:", A, file=fw)
    print ("T:", T, file=fw)
    print ("Use prior:", use_prior, file=fw)
    print ("Zero diagonal:", non_diagonal, file=fw)
    print ("min # of msgs:", min_number_msg, file=fw)
    print ("merge_consecutive_from_same_sender:", merge_consecutive_from_same_sender, file=fw)
    print ("remove_stop_words:", remove_stop_words, file=fw)
    print ("time decay gamma prior", time_decay_prior, file=fw)
    print ("influence gamma prior", influence_prior, file=fw)
    print ("word concentration gamma prior", word_concentration_prior, file=fw)
    print ("word pseudocount gamma prior", word_pseudocount_prior, file=fw)

    fw.close()

    # mcmc sample
    # get the samples
    log_prior_and_log_likelihood_SAMPLE, time_decays_SAMPLE, influences_SAMPLE, word_concentrations_SAMPLE, word_pseudocounts_SAMPLE = sample_bec(bec, display=display,
                                                                                                                                                  sample_number=sample_number, burn_in_number=burn_in_number, resultDirName=resultDirName)

    print ("Results written to", resultDirName)

    # setting the bec parameters to be posterior mean
    # and then compute the evolution of base measure and excitation
    # pseudocounts

    print ("Computing the estimated base measure and excitation pseudocounts from posterior median...")

    # save the selected tokens
    tokens_selected = talkbankxml_instance.getSelectedTokens()
    fw = open(resultDirName + "tokens-selected.txt", "w")
    print ("token.id\ttoken", file=fw)
    for j, w in enumerate(tokens_selected):
        w = w.encode("ascii", "ignore")
        print ("%d\t\"%s\"" % (j + 1, w), file=fw)
    fw.close()

    # posterior * median * of the parameters
    time_decay_POST_MEAN = median(time_decays_SAMPLE, axis=0)
    word_pseudocounts_POST_MEAN = median(word_pseudocounts_SAMPLE, axis=0)
    influences_POST_MEAN = median(influences_SAMPLE, axis=0)
    word_concentrations_POST_MEAN = median(word_concentrations_SAMPLE, axis=0)

    # set the estimate to bec parameters
    bec.set_time_decay(time_decay_POST_MEAN, x=None)
    bec.set_word_concentration(word_concentrations_POST_MEAN, x=None)
    bec.set_word_pseudocounts(word_pseudocounts_POST_MEAN, x=None, v=None)
    for x in range(A):
        for y in range(A):
            if not x == y:
                bec.set_influence(influences_POST_MEAN[x, y], x, y)

    # compute the evolution of base measure and excitation pseudocounts
    base_measure_array = zeros((len(allMessages), V))
    weighted_excitation_pseudocounts_array = zeros((len(allMessages), V))

    for i, msg in enumerate(allMessages):
        base_measure_array[i, :] = msg.base_measure()
        weighted_excitation_pseudocounts_array[
            i, :] = msg._influence.dot(msg._excitation_pseudocounts)

    # write to file
    savetxt(resultDirName + "base_measure.txt", base_measure_array,
            header=v_header, fmt="%.7f", comments='')
    savetxt(resultDirName + "weighted_excitation_pseudocounts.txt",
            weighted_excitation_pseudocounts_array, header=v_header, fmt="%.7f", comments='')

    print ("Base measure & excitation evolution saved to", resultDirName)


if __name__ == "__main__":
    main()
