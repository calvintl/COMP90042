# Calvin Tanujaya Lim
# 355141
# Project 1

from __future__ import division
from nltk.corpus import comtrans
from collections import defaultdict
import random

'''
Activity Log

1st hour - 2nd hour

For the first couple of hours I spent my time reading the koehn book, starting
at chapter 4.4, which is about the higher IBM Models. I read through till
the end of IBM Model 5 (chapter 4.4.5). However, I spent more time trying to
grasp the stuff from chapter 4.4.1 (IBM Model 2) and 4.4.2 (IBM Model 3),
thinking to try and implement these models for the project.

What I learned from reading the material is the difference between each
of the IBM Models, and how it improved with each increment of the model.

I also noted what the different steps within the IBM models meant. How
IBM Model 2 has an extra alignment step to find the most probable positions
for each of the translated words. In IBM Model 3, it has 2 more steps compared
to IBM Model 2, the NULL insertion step and the fertility step. NULL insertion
is what it sounds like, to insert NULL tokens to account for words that has
no correspondent in the input. The fertility step is to create more duplicates
of a certain word, if that word usually translates to more than one word. The
amount of duplicates depend on how many words it usually translates into.

3rd hour - 9th hour

After reading the materials, I started to implement IBM Model 2. With the help
of the pseudocode from the koehn book, implementing it in Python didn't took
very long, around 3 hours at most. Which includes fixing errors such as
forgetting to import the required stuff (ie. defaultdict, division),
not initializing the variables properly, also deciding the data structures
for the variables in the algorithm.

I decided to use a 4-dimension dictionary for the align data structure. Seeing
as how the t(e|f) was a 2-d dictionary, I thought that making the align data
structure to be modelled similar to that is quite easy.

What took more time was to actually understand what the
code does and why is it doing what it's doing. For example, I spent some time
researching on the internet as to why the value 'j' starts at 1 instead of 0,
however I couldn't find any resources that relates as to why it was like that.
The value of 'j' is used as the word positions for the english text.

Running the code over the set of very very small sentences (ie. the set of
sentences used to test for IBM Model 1 in worksheet 5), it just doesn't
produce an output that makes sense. It is because that the value 'j' starts
at 1 that it skips the 1st english word in each of the sentence pair. As a
result this creates the wrong lexical translation probabilities, and in the
end, the wrong alignment probabilities.

Understanding the mathematical aspect of the model also helps alot in
understanding the code. I had to learn what is 's-total' in the pseudocode,
and what 'c' is. Although I'm still vague as to what they are, but what
s-total doing is, for a certain english word, it is summing up the
multiplication of the translation probability of that word to a foreign word
and the probability that it is aligned to that position. So it's like the total
of all possible translation and alignment probability of that particular english
word. This value is then used in calculating 'c', which is to divide the
(translation prob * alignment prob) by s-total, which gives us the value
of it's probability happening.

10th hour - 12th hour

I started trying to implement IBM Model 3. At first, I read the pseudocode
and try to make a sense out of it. Tried to understand what's it's doing
but it was a little too complicated, so I had to sort of re-read the chapter
4.4.2. There are plenty of new concepts starting in IBM Model 3. Such as
NULL insertion, fertility, as well as sampling and hill climbing.

Looking at the pseudocode, I decided to start implementing the smaller
functions first. So I started at the function sample(e, f), then I got stuck
at line 5 in the function pseudocode, which goes a(j') = argmaxj' t(ej'|fi')
d(i'|j', length(e), length(f)). It took me quite a while understanding
what that line was meant to do. I wondered where i' was declared and I had to
do some research regarding argmax. After spending sometime figuring out, I
finally understood that the line was to find i' which maximizes the probability
of both the translation and distribution probability.

So then I started to code it in Python, without testing it yet.

I still haven't finish the function sample(e,f) as it requires the other 2
small functions hillclimb(a,j) and neighboring(a,j) to be implemented as well.

13th hour - 17th hour

Before implementing hillclimb(a,j), I saw that it requires the neighboring
function as well. So I read page 104 in Koehn book chapter 4, just to refresh
my memory what neighboring alignments meant. What it means when an alignment
differ by a move or a swap.

And again, I tried to understand what the code was doing. So the neighboring
function returns N, which is the set of neighboring alignments given an
alignment a. And if the probability of the neighboring alignment is higher
than the current alignment, then we set the new a as the a_neighbor. We
have to repeat this process until it doesn't changes the a anymore, which
means that it has found the maximum alignment probability. Having understood
what it was suppose to do, I tried implementing it.

There was a problem though with trying to implementing the probability(a)
function. I didn't know how to go about it and in the Koehn book chapter 4,
page 105, it said that the function follows straightforwardly from Equation 4.33
. I had a look at the equation and it was really complicated. After spending
quite a lot of time without making much progress, I decided to stop implementing
IBM Model 3.

I did have a look at the giza++ code and wasn't really much help. It could be
because of the unfamiliarity with the language but I thought that the algorithm
in the giza++ model doesn't look the same as the one in Koehn book. I don't
think the giza++ model used any sampling of the sentence pairs. The giza++
code also had more complicated probability calculations than I expected.

18th hour - 22th hour

After stopping IBM Model 3, I decided to do more extensive testing on my IBM
Model 2, because I didn't really do much testing on it at the time I finished
the implementation.

Doing testing, I think the starting 'j' value is really causing alot of problems
for me here. As mentioned before because the 'j' value starts at 1, it skips
the first english word, positioned 0 in the list, and does all the calculations.

After more testing, it seems that adding the None token to the foreign sentence
as well as extending the range of l_e to l_e+1 and when indexing the english
word we need to make sure that it's indexing the right position. In this case
the position of the english words will be j-1 because of j starting at 1.

I then did testing using sentences from nltk.corpus, the comtrans. I used only
a subset of the sentences, about 20-50, and then compare the alignments. At
first I just did it manually by checking each alignment. But it gets really
tedious for sentences that are really long.

From testing, it also reveals that this implementation rely heavily on the
lexical translation step. Without enough data to have a good t_ef table,
the alignment step will be absolutely useless. It will have all the wrong
alignments or even having the same probability for all possible positions in
the sentence. So it is hard to achieve a high accuracy/precision. This is based
on testing my implementation using the comtrans sentences. The accuracy
for the final alignment tends to be pretty bad, but that is because I've
only used the first 30 sentences out of the comtrans. It may not be enough
to create a concrete and good translation table. Some alignments are correct
but most of them aren't. I couldn't do alot sentences as it crashes most of the
time while processing.

Also, while testing, it has come to my attention that the value for
i and j in the alignment table (a[i][j][l_e][l_f]) starts at 1 instead of
0, just according to the example in the Koehn book page 98 chapter 4.

I created a function test_IBMModel2() which tests my IBM model 2 implementation
against the data from the corpus comtrans. I compared the alignment from using
the .precision function in the corpus and return the average precision. As
expected, I had a really low precision. As with my justification earlier about
having little amount of sentences the precision goes down as well. With
greater amount, the precision gets better. About 0.10 avg precision using
just 10 sentences, and about 0.32 when using 40 sentences.

23th - 24th hour
The last couple of hours I was just documenting the code as well as reading
through my activity log, making sure that everything has been covered.
'''

sent = [(['the','house'], ['das','haus']),
        (['the','book'], ['das','buch']),
        (['a', 'book'], ['ein', 'buch']),
        (['a', 'house'], ['ein', 'haus'])]

sent1 = [(['the','house'], ['das','haus']),
        (['the','book'], ['das','buch']),
        (['a', 'book'], ['ein', 'buch'])]

sent2 = [(['the','house'], [None,'das','haus']),
        (['the','book'], [None,'das','buch']),
        (['a', 'book'], [None,'ein', 'buch']),
        (['of','course'], [None, 'naturlich']),
        (['of', 'course', 'the', 'house', 'is', 'small'],
         [None,'naturlich', 'ist', 'das', 'haus', 'klein'])]

def ibm_model1(sentence_pairs, iterations):

    '''
    Testing ibm_model1 according to the example from the koehn book chapter 4
    figure 4.4

    >>> print "%.4f" % ibm_model1(sent1, 3)['the']['das']
    0.7479

    >>> print "%.4f" % ibm_model1(sent1, 3)['book']['das']
    0.1208

    >>> print "%.4f" % ibm_model1(sent1, 3)['house']['das']
    0.1313

    >>> print "%.4f" % ibm_model1(sent1, 3)['the']['buch']
    0.1208
    
    >>> print "%.4f" % ibm_model1(sent1, 3)['book']['buch']
    0.7479
    
    >>> print "%.4f" % ibm_model1(sent1, 3)['a']['buch']
    0.1313
    
    >>> print "%.4f" % ibm_model1(sent1, 3)['book']['ein']
    0.3466
    
    >>> print "%.4f" % ibm_model1(sent1, 3)['a']['ein']
    0.6534
    
    >>> print "%.4f" % ibm_model1(sent1, 3)['the']['haus']
    0.3466

    >>> print "%.4f" % ibm_model1(sent1, 3)['house']['haus']
    0.6534
    '''

    # Get all the words
    
    eng_words = []
    foreign_words = []

    for (eng, foreign) in sentence_pairs:
        for word in eng:
            if word not in eng_words:
                eng_words.append(word)

        for word in foreign:
            if word not in foreign_words:
                foreign_words.append(word)

    initial_probability = 1 / len(eng_words)

    table_ef = defaultdict(lambda: defaultdict(lambda: initial_probability))
    count_ef = defaultdict(lambda: defaultdict(float))
    total_f = defaultdict(float)
    s_total = defaultdict(float)

    iteration = 0
    while(iteration < iterations):

        #Initialize
        i = 0
        while (i < len(eng_words)):
            j = 0
            while (j < len(foreign_words)):
                count_ef[eng_words[i]][foreign_words[j]] = 0
                total_f[foreign_words[j]] = 0
                j+=1
            i+=1

        for (eng, foreign) in sentence_pairs:
            for word in eng:
                s_total[word] = 0
                for for_word in foreign:
                    s_total[word] += table_ef[word][for_word]

            for word in eng:
                for for_word in foreign:
                    count_ef[word][for_word] += (table_ef[word][for_word] /
                                                 s_total[word])
                    total_f[for_word] += (table_ef[word][for_word] /
                                          s_total[word])

        for for_word in foreign_words:
            for word in eng_words:
                table_ef[word][for_word] = (count_ef[word][for_word] /
                                            total_f[for_word])
                

        iteration += 1

    return table_ef

        
def ibm_model2(sentence_pairs, iterations):

    table_ef = ibm_model1(sentence_pairs, iterations)
    
    a = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(float))))

    eng_words = []
    foreign_words = []

    for (eng, foreign) in sentence_pairs:
        for word in eng:
            if word not in eng_words:
                eng_words.append(word)

        for word in foreign:
            if word not in foreign_words:
                foreign_words.append(word)

    #Initialize alignment

    for (eng, foreign) in sentence_pairs:
        l_e = len(eng)
        l_f = len(foreign)
    
        i = 0
        while(i < l_f):
            j = 1
            while(j < l_e+1):
                a[i][j][l_e][l_f] = 1 / (l_f + 1)
                j += 1
            i += 1
    
    iteration = 0
    while(iteration < iterations):

        # initialize
        count_ef = defaultdict(lambda: defaultdict(lambda: 0.0))
        total_f = defaultdict(lambda: 0.0)
        count_a = defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
        total_a = defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: 0.0)))        
        s_total = defaultdict(float)
        
        for (eng, foreign) in sentence_pairs:
            l_e = len(eng)
            l_f = len(foreign)
            
            # compute normalization
            for j in range(1, l_e+1):
                en_word = eng[j-1]
                s_total[en_word] = 0
                
                for i in range(0, l_f):
                    s_total[en_word] += (table_ef[en_word][foreign[i]] *
                                         a[i][j][l_e][l_f])

            # collect counts
            for j in range(1, l_e+1): 
                en_word = eng[j-1]

                for i in range(0, l_f):
                    for_word = foreign[i]
                    
                    c = (table_ef[en_word][for_word] * a[i][j][l_e][l_f] /
                         s_total[en_word])
                    count_ef[en_word][for_word] += c
                    total_f[for_word] += c
                    count_a[i][j][l_e][l_f] += c
                    total_a[j][l_e][l_f] += c

        # estimate probabilities
        table_ef = defaultdict(lambda: defaultdict(lambda: 0.0))
        a = defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda:defaultdict(lambda: 0.0))))

        for word in eng_words:
            for for_word in foreign_words:
                table_ef[word][for_word] = (count_ef[word][for_word] /
                                            total_f[for_word])

        for (eng, foreign) in sentence_pairs:
            l_e = len(eng)
            l_f = len(foreign)
            
            for j in range(1, l_e+1):
                for i in range(0, l_f):
                    a[i][j][l_e][l_f] = (count_a[i][j][l_e][l_f] /
                                         total_a[j][l_e][l_f])

        iteration += 1
    return a, table_ef

def test_IBMModel2():

    no_of_sentences = 30
    
    sentences = comtrans.aligned_sents()[:no_of_sentences]
    sent_pairs = []

    # Adding the None values to the foreign sentences and constructing them
    # into the correct format to use with the ibm_model2()
    for sentence in sentences:
        eng_words = sentence.mots
        foreign_words = [None] + sentence.words
        sent_pairs.append((eng_words,foreign_words))

    align, t_ef = ibm_model2(sent_pairs, 15)

    fin_align = []  # The list of final alignments

    # Finding the best alignment for each of the words in the sentences
    for (e, f) in sent_pairs:
        l_e = len(e)
        l_f = len(f)
        curr_align = []
        for i in range(1, l_e+1):
            max_prob = -1
            for j in range(1, l_f+1):
                prob = align[j][i][l_e][l_f]
                if max_prob < prob:
                    max_prob = prob
                    max_j = j
            curr_align.append((i-1, max_j-1))
            
        fin_align.append(curr_align)

    # Calculating the precision of the alignments
    avg_precision = 0
    count = 0
    for sent_alignments in fin_align:
        algn = ''
        for (e, f) in sent_alignments:
            algn += "%d-%d " %(f,e)

        avg_precision += sentences[count].precision(algn)
        count += 1

    avg_precision /= count

    return avg_precision
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()




#####################################################################
# IBM Model 3 : INCOMPLETE - Can't run it because it's not complete #
#####################################################################

'''
def ibm_model3(sentence_pairs, iterations):

    d, t_ef = ibm_model2(sentence_pairs, 10)

    iteration = 0
    while(iteration < iterations):

        count_t = defaultdict(lambda: defaultdict(lambda: 0.0))
        count_d = defaultdict(lambda: defaultdict(lambda: 0.0))
        count_f = defaultdict(lambda: defaultdict(lambda: 0.0))
        count_p1 = 0
        count_p0 = 0

        total_t = defaultdict(lambda: 0.0)
        total_d = defaultdict(lambda: defaultdict(lambda: defaultdict(
            lambda: 0.0)))
        total_f = defaultdict(lambda: 0.0)

        for (eng, foreign) in sentence_pairs:
            A = sample(eng, foreign, t_ef, d)

            c_total = 0
    
    return 0


def sample(e, f, t_ef, d):

    a = defaultdict(float)
    
    l_f = len(f)
    l_e = len(e)
    for j in range(0, l_f):
        for i in range(1, l_e):
            a[j] = i # pegging one alignment point
            for j_inv in range(0, l_f):
                if j_inv != j:
                    a[j_inv] = argmax(e, f, j_inv, t_ef, d)

            new_a = hillclimb(a, j, e, f, d)

            # add neighboring(a,j) to set A

def hillclimb(a, j_pegged, e, f, d):
    
    while True:
        a_old = a
        for a_neighbor in neighboring(a, j_pegged):
            
    
    return a
            
def argmax(e, f, j_inv, t_ef, d):
    max_trans = -1.0
    max_trans_index = []

    final_i = None
    
    # find the most probable translation 1st
    for i in range(0, len(f)):
        prob = t_ef[e[j_inv]][f[i]]
        if prob > max_trans:
            max_trans = prob
            max_trans_index.append(i)

        # if the word has a probability of translating to another word
        # and has equal probability to previous translations
        elif prob == max_trans and max_trans_index != None:
            max_trans_index.append(i)

    max_prob_align = -1.0

    l_e = len(e)
    l_f = len(f)
            
    for idx in max_trans_index:
        prob_align = d[idx][j_inv][l_e][l_f]
        if prob_align > max_prob_align:
            max_prob_align = prob_align
            final_i = idx
            
    return final_i
'''
