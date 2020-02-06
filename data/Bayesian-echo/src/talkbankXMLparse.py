# the parser class for talkbank's XML data
from numpy import bincount, array, polyfit, poly1d, linspace
from bec import Message  # Message objects with start and end time
import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer
import operator
import matplotlib.pyplot as plt


class talkbankXML(object):
    """docstring for talkbankXML"""
    def __init__(self, xml_filename, remove_stop_words=True, min_number_msg=10, merge_consecutive_from_same_sender=True,language='eng'):
        '''
        Initialize xml parser object.

        xml_filename: path of the xml talkbank data file

        remove_stop_words: remove stop words or not
        '''
        self._xml_filename = xml_filename
        self._tree = ET.parse(self._xml_filename)
        self._root = self._tree.getroot()
        self._stemmer = PorterStemmer()
        self._language = language

        # load stop words
        self.loadStopWords(self._language)

        # participants
        self._participants = {} # id -> {0,1,2,..,A-1} dict of all participants
        self._participant_list = [] # id's corresponding to {0,1,...,A-1}
        self._A = 0
        for participant_entry in self._root[0]:
            tmp = participant_entry.tag.split("}")
            assert tmp[-1]=="participant"
            self._participants[participant_entry.attrib["id"]] = self._A
            self._participant_list.append(participant_entry.attrib["id"])
            self._A += 1

        print ("%d participants" % (self._A) )
        print ("Reading messages...")
        if merge_consecutive_from_same_sender: print ("* will merge two consecutive messages from the same sender *")
        # messages
        self._msg_number = 0
        self._all_word_count = {}
        self._all_stop_word_count = {}
        # all messages indexed by the sender, in the form of a list of tokens, tokens not selected
        self._all_messages_raw = {} 
        self._all_messages_start_times_raw = {}
        self._all_messages_end_times_raw = {}
        for x in range(self._A):
            self._all_messages_raw[x] = []
            self._all_messages_start_times_raw[x] = []
            self._all_messages_end_times_raw[x] = []

        last_sender = None
        for msg_entry in self._root:
            if not msg_entry.tag.split('}')[-1]=="u":
                continue
            msg = []
            start_time = None
            end_time = None
            # words in msg
            for w in msg_entry:
                if w.tag.split("}")[-1]=="w":
                    # this is a word
                    this_word = w.text.rstrip()
                    this_word = this_word.lstrip()
                    # lower-case only
                    this_word = this_word.lower()
                    # stemming into standard form
                    this_word = self._stemmer.stem(this_word)
                    # remove stop word
                    if remove_stop_words:
                        if this_word in self._stop_word_set:
                            try:
                                self._all_stop_word_count[this_word] += 1
                            except:
                                self._all_stop_word_count[this_word] = 1
                        else:
                            msg.append(this_word)
                            try:
                                self._all_word_count[this_word] += 1
                            except:
                                self._all_word_count[this_word] = 1
                    else:
                        msg.append(this_word)
                        try:
                            self._all_word_count[this_word] += 1
                        except:
                            self._all_word_count[this_word] = 1                     

                elif w.tag.split("}")[-1]=="media":
                    # this is media info
                    start_time = float(w.attrib["start"])
                    end_time = float(w.attrib["end"])
                    assert end_time > start_time, "Found start_time %s >= end_time %s!" % (start_time, end_time)

            if len(msg)>0 and (start_time is not None) and (end_time is not None):
                # add this message
                self._msg_number += 1
                # if self._msg_number == 100: break # break #
                sender = self._participants[msg_entry.attrib["who"]]
                # if the time is the same as the last one, merge this message with the last msg !!!
                # * IMPORTANT *, otherwise duplicates in timestamps would mess up the time generation process
                # because the times stampes are unique with prob 1
                
                # also, 
                # if merge_consecutive_from_same_sender=True, 
                # merge two consecutive utterance from the same sender
                if (len(self._all_messages_end_times_raw[sender])>0 and start_time == self._all_messages_end_times_raw[sender][-1]) or (merge_consecutive_from_same_sender and (last_sender is not None) and sender==last_sender):
                    for wd in msg:
                        self._all_messages_raw[sender][-1].append(wd)
                        # modify (t1, t2) + (t2, t3) to (t1, t3)
                        self._all_messages_end_times_raw[sender][-1] = end_time
                else:
                    self._all_messages_raw[sender].append(msg)
                    self._all_messages_start_times_raw[sender].append(start_time)
                    self._all_messages_end_times_raw[sender].append(end_time)

                last_sender = sender

        # remove agents who do not speak
        A_new = 0
        numbering_mapping = {}  # mapping: old agent numbering -> new agent numbering
        inverse_numbering_mapping = {} # mapping: new agent numbering -> old agent numbering
        for x in sorted(self._all_messages_raw.keys(), key=lambda x:len(self._all_messages_raw[x]), reverse=True):
            if len(self._all_messages_start_times_raw[x])<min_number_msg:
                print ("Removed: agent", x, "%d messages" % (len(self._all_messages_start_times_raw[x])))
            else:
                # a speaker
                numbering_mapping[x] = A_new
                inverse_numbering_mapping[A_new] = x
                A_new += 1
        # update
        self._A = A_new
        all_messages_raw_updated = {}
        all_messages_start_times_raw_updated = {}
        all_messages_end_times_raw_updated = {}

        participant_list = []
        participants = {}
        for x in range(A_new):
            participant_list.append(self._participant_list[inverse_numbering_mapping[x]])
            participants[self._participant_list[inverse_numbering_mapping[x]]] = x
        self._participant_list = participant_list
        self._participants = participants
        for x in self._all_messages_raw.keys():
            if x in numbering_mapping.keys():
                all_messages_raw_updated[numbering_mapping[x]] = self._all_messages_raw[x][:]
                all_messages_start_times_raw_updated[numbering_mapping[x]] = self._all_messages_start_times_raw[x][:]
                all_messages_end_times_raw_updated[numbering_mapping[x]] = self._all_messages_end_times_raw[x][:]

        self._all_messages_raw = all_messages_raw_updated
        self._all_messages_start_times_raw = all_messages_start_times_raw_updated
        self._all_messages_end_times_raw = all_messages_end_times_raw_updated


        if remove_stop_words:
            print ("Stop tokens removed")
            print ("Processed %d messages: %d non-stop token types appeared, %d stop token types appeared" % (self._msg_number, len(self._all_word_count.keys()), len(self._all_stop_word_count.keys())))
        else:
            print ("Stop tokens reserved")
            print ("Processed %d messages: %d token types appeared" % (self._msg_number, len(self._all_word_count.keys()))       )

    def get_participants(self):
        ''' return a dict: name -> {0,...,A-1}'''
        return self._participants

    def get_participant_list(self):
        ''' return the list of participants in terms of name'''
        return self._participant_list

    def save_cast_table(self, result_dir):
        ''' save a table of cast (mapping of agent # and id) to a dir'''
        fw = open(result_dir + "cast.txt", 'w')
        print ("agent.num\tagent.name",file=fw)
        for i, x in enumerate(self._participant_list):
            print ('''%d\t"%s"''' % (i+1, x),file=fw)
        fw.close()

    def loadStopWords(self,language):
        self._stop_word_set = set() 
        if language=='chinese':
            stop_word_list_filename = "./stopwords/chinese.stop"
        else:
            stop_word_list_filename = "./stopwords/english.stop"

        fr = open(stop_word_list_filename, 'r')
        for l in fr:
            l = l.rstrip()
            l = l.lstrip()
            if language=='chinese':
                l = l.decode('utf-8')
            w = self._stemmer.stem(l)
            self._stop_word_set.add(w)
        fr.close()
        print (len(self._stop_word_set), "stop words loaded.")

    def exportMessages(self, V, maxTime=100.0, display=0):
        """
        Export as a list of messageLists, where each messageList is a list of Message objects with the same sender.

        V: effective size of vocabulary, chosen to be the 1 ~ V most frequent tokens among all tokens

        maxTime: the time would be normalized so that the lastest message has the timestamp as maxTime
        """
        if V > len(self._all_word_count.keys()):
            print ("\nWarning: V = %d > %d actual tokens" % (V, len(self._all_word_count.keys())))
        # assert V <= len(self._all_word_count.keys())
        self._V = V
        self._vocab = {}    # selected tokens, mapping from V tokens to 0,1,..,V-1
        self._vocab_list = []   # selected tokens, but stored as a list of size V
        cnt = 0 
        exportedMsgCount = 0
        for token in sorted(self._all_word_count, key=self._all_word_count.get, reverse=True):
            self._vocab[token] = cnt
            self._vocab_list.append(token)
            cnt += 1
            if cnt>=V: break
        if display>0:
            for token in self._vocab_list:
                print ("%s: %d occurrences" % (token, self._all_word_count[token]))
        listOfMSGLists = []
        listOfMSG_StartTimeLists = []
        listOfMSG_EndTimeLists = []

        maxActualTime = 0
        w_count = 0
        for x in range(self._A):
            msgList = []    # list of messages from sender x, in the form of counts, tokens selected
            startTimeList = []
            endTimeList = []
            for i, msg in enumerate(self._all_messages_raw[x]):
                start_time = self._all_messages_start_times_raw[x][i]
                end_time = self._all_messages_end_times_raw[x][i]
                assert end_time > start_time, "Found start_time %s >= end_time %s!" % (start_time, end_time)

                msg_converted = []
                for w in msg:
                    w_count += 1
                    try:
                        msg_converted.append(self._vocab[w])
                    except:
                        continue
                if len(msg_converted)>0:
                    msgList.append(bincount(msg_converted, minlength=self._V))
                    exportedMsgCount += 1
                    if end_time>maxActualTime:
                        maxActualTime = end_time
                    startTimeList.append(start_time)
                    endTimeList.append(end_time)
                    
            
            listOfMSGLists.append(msgList)
            listOfMSG_StartTimeLists.append(startTimeList)
            listOfMSG_EndTimeLists.append(endTimeList)
        
        if maxTime is None:
            maxTime = maxActualTime
        
        if display>1:
            for x in range(self._A):
                print ("Messages from sender", x)
                for i, msg in enumerate(listOfMSGLists[x]):
                    print (msg)
                    print ("start time", listOfMSG_StartTimeLists[x][i] * 1.0 /maxActualTime * maxTime)
                    print ("end time", listOfMSG_EndTimeLists[x][i] * 1.0 /maxActualTime * maxTime)


        # convert to Message objects 
        allMessages = []    # the list of all messageLists, of length A
        for x in range(self._A):
            messageList = []
            for i, msg in enumerate(listOfMSGLists[x]):
                start_time = listOfMSG_StartTimeLists[x][i]/maxActualTime * maxTime
                end_time = listOfMSG_EndTimeLists[x][i]/maxActualTime * maxTime

                messageList.append(Message(sender=x, sender_name=self._participant_list[x],
                    start_time=start_time, end_time=end_time, 
                    A=self._A, 
                    V=self._V, 
                    T=maxTime,
                    tokens=None, # it is not for token types, but for raw msg
                    token_type_counts=msg))
            allMessages.append(messageList)

        print ("%d messages exported" % (exportedMsgCount))
        for x in range(self._A):
            print ("%d messages from agent %d: %s" % (len(allMessages[x]), x, self._participant_list[x]))

        print ("\nTotal token counts:", sum([msg.get_total_token_counts() for x in range(self._A) for msg in allMessages[x]]))
        print ("Total token counts in raw text without restricting to top-V", w_count)
        print ("Removal rate = %.2f%%" % ((1-sum([msg.get_total_token_counts() for x in range(self._A) for msg in allMessages[x]]) * 1.0 / w_count)*100))

        return allMessages


    def exportToLDAC(self, allMessages, time_to_split, fileNamePrefix="./"):
        '''
        save an allMessages object as LDAC format, which is for input to Blei's topic model programs.

        time_to_split: messages with t>time_to_split go to the test set.
        All previous messages are put into time slices such that each time slice contains the same number 
        of messages as test_set_size
        '''
        # filenames
        multFileName = fileNamePrefix + "-mult.dat"
        seqFileName = fileNamePrefix + "-seq.dat"
        infoFileName = fileNamePrefix + "-info.dat"

        # mult file
        fw1 = open(multFileName, 'w')
        # a list all messages
        messages = [msg for messageList in allMessages for msg in messageList]
        messages = sorted(messages, key=lambda msg: msg.get_start_time())
        V = messages[0].get_V()

        print ("\n%d messages in total" % (len(messages)))
        maxTime = max((msg.get_end_time() for msg in messages))
        print ("maxTime =", maxTime)
        print ("V =", V)

        # unique_word_count index1:count1 index2:count2 ... indexn:counnt
        for msg in messages:
            token_type_counts = msg.count_token_types().tolist()[:]
            writeStr = ""
            unique_word_count = 0
            for w in range(V):
                if token_type_counts[w]>0:
                    unique_word_count += 1
                    writeStr += " %d:%d" % (w, token_type_counts[w])
            writeStr = "%d" % (unique_word_count) + writeStr
            # print token_type_counts
            print (writeStr,file=fw1)

        fw1.close()
        print ("Mult file saved to", multFileName)

        # seq file
        # equal size split
        test_set_size = len([msg for msg in messages if msg.get_start_time()>time_to_split])
        print (test_set_size)
        print (len(messages) * 1.0 / test_set_size)
        # ensure that the last split has exactly test_set_size # of messages
        number_time_slices = int(round(len(messages)*1.0/test_set_size))
        number_docs_times = [0] * (number_time_slices-1) + [test_set_size]
        for j in range(number_time_slices-1):
            number_docs_times[j] = int((len(messages)-test_set_size)*1.0/(number_time_slices-1))
        number_docs_times[0] = len(messages) - test_set_size - sum(number_docs_times[1:-1])
        # ensure that the total number of messages preserved
        assert(sum(number_docs_times)==len(messages))
        print ("Training set size = %d, test set size = %d" % (sum(number_docs_times[0:-1]), test_set_size))

        # print sum(number_docs_times)
        fw2 = open(seqFileName, 'w')
        print ( "%d" % (len(number_docs_times)),file=fw2)
        for x in number_docs_times:
            print ("%d" % (x),file=fw2)
        fw2.close()
        print ("%d time windows splitted" % (len(number_docs_times)))
        print ("Seq file saved to", seqFileName)

        # # vocab file
        # fw3 = open(vocabFileName, 'w')
        # for v in self._vocab_list:
        #   print >>fw3, v
        # fw3.close()
        # print "Tokens saved to", vocabFileName

        # info file
        fw4 = open(infoFileName, 'w')
        for msg in messages:
            print ("%d %.3f %d" % (msg.get_sender(), msg.get_start_time(), sum(msg.count_token_types())),file=fw4)
        fw4.close()
        print ("Info file saved to", infoFileName)

        print ("\nLDAC exported")

        return number_docs_times

    def V(self):
        return self._V

    def A(self):
        return self._A

    def participants(self):
        return self._participants

    def allWordCount(self):
        return sorted(self._all_word_count.iteritems(), key=operator.itemgetter(1), reverse=True)

    def rawTokenNumber(self):
        return len(self._all_word_count.keys())

    def getSelectedTokens(self):
        return self._vocab_list[:]


def plotAllMessages(allMessages):
    '''Plotting a list of MessageLists. 
    Each agent is shown with a barplot with each bar span (start_time, end_time) and its height 
    being the total number of tokens in that utterance.'''
    A = allMessages[0][0].get_A()
    T = allMessages[0][0].get_T()
    assert A==len(allMessages)

    fig = plt.figure()
    for x in range(A):
        # bar plot
        ax = fig.add_subplot(A,1,x+1)
        agent_name = allMessages[x][0].get_sender_name()
        start_times_vec = array([msg.get_start_time() for msg in allMessages[x]])
        end_times_vec = array([msg.get_end_time() for msg in allMessages[x]])
        duration_vec = end_times_vec - start_times_vec
        volume_vec = array([msg.get_total_token_counts() for msg in allMessages[x]])
        ax.bar(start_times_vec, volume_vec, duration_vec, color='y')
        ax.set_title("agent %s (%d utterances)" % (agent_name, len(allMessages[x])))
        ax.set_xlim(0, T)

    plt.tight_layout(pad=1, h_pad=1, w_pad=1)

    fig2 = plt.figure()
    for x in range(A):
        # width ~ total count of tokens
        ax = fig2.add_subplot(4,3,x+1)
        agent_name = allMessages[x][0].get_sender_name()
        start_times_vec = array([msg.get_start_time() for msg in allMessages[x]])
        end_times_vec = array([msg.get_end_time() for msg in allMessages[x]])
        duration_vec = end_times_vec - start_times_vec
        volume_vec = array([msg.get_total_token_counts() for msg in allMessages[x]])
        linear_fit = polyfit(volume_vec, duration_vec, 1)
        fitted_fun = poly1d(linear_fit)

        x_prime = linspace(0, max(volume_vec), 100)
        ax.plot(volume_vec, duration_vec, "kx")
        ax.plot(x_prime, fitted_fun(x_prime), "r--")
        ax.set_title("\nagent %s t = %.4f v + %.2f\n" % (agent_name, linear_fit[0], linear_fit[1]))
        ax.set_xlabel("# of tokens in utterance")
        ax.set_ylabel("Duration")

    plt.tight_layout(pad=1, h_pad=1, w_pad=1)
    plt.show()


if __name__=="__main__":
    
    xml_filename = 'test.xml'#"../data/12-angry-men/12-angry-men.xml"


    talkbankxml_instance = talkbankXML(xml_filename, 
        remove_stop_words=False, 
        min_number_msg=10, 
        merge_consecutive_from_same_sender=True,
        language='eng')
    
    # print talkbankxml_instance.participants()

    # print talkbankxml_instance.allWordCount()

    allMessages = talkbankxml_instance.exportMessages(V=20, maxTime=100.0, display=1)

    plotAllMessages(allMessages)
