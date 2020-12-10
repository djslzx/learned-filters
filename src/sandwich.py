import torch as T
from bloom import Bloom, WordBloom
from model import WordNet
from math import log, log2
from util import ilen

class Sandwich:

    def __init__(self, n, c, err, set_size, err1k):
        """
        n: number of letters in string
        c: size of alphabet
        err: total error rate of sandwich
        """
        self.n = n
        self.c = c

        self.model = WordNet(n, c)
        self.tau = 0.5 # default value, adjust by tuning later
        self.alpha = 0.618503137801576 # 2 ** -log(2)

        # AMQs can only be set up after training model
        self.err = err
        self.err1 = self.err * err1k
        self.amq1 = WordBloom(Bloom.init_ne(set_size, self.err1))
        self.amq2 = None # Determine size after training

    def _choose_tau(self, xs, ys, taus=T.arange(0,1,0.1)):
        """
        Measure false positive rate of model on xs,ys 
        using taus as thresholds
        """
        negatives = [x for x,y in zip(xs,ys) if not y]
        positives = [x for x,y in zip(xs,ys) if y]
        prediction = {x:self.model(x) for x in xs}

        num_neg = len(negatives)
        num_pos = len(positives)

        def fpr(tau):
            """fpr = false pos / total neg"""
            return ilen(x for x in negatives 
                        if prediction[x] > tau)/num_neg

        def fnr(tau):
            """fnr = false neg / total pos"""
            return ilen(x for x in positives
                        if not (prediction[x] > tau))/num_pos

        # Choose the tau that minimizes fnr,
        # with constraint that fpr(tau) <= err
        best_fpr_tau = taus[0]
        best_fpr = fpr(taus[0])
        candidates = [taus[0]] if best_fpr < self.err/2 else []

        for tau in taus[1:]:
            fp_rate = fpr(tau)
            if fp_rate < best_fpr:
                best_fpr_tau = tau
                best_fpr = fp_rate
            if fp_rate < self.err/2:
                candidates.append(tau)

        print("candidates:", candidates)

        # If no tau has fpr < err/2, choose tau with best fpr
        if not candidates:
            print("tau={}, fpr={}, fnr={}".format(best_fpr_tau, best_fpr, fnr(best_fpr_tau)))
            return best_fpr_tau, fpr(best_fpr_tau), fnr(best_fpr_tau)
        # Otherwise, choose tau in candidates with best fnr
        else:
            best_fnr_tau = candidates[0]
            best_fnr = fnr(candidates[0])
            for tau in candidates:
                fn_rate = fnr(tau)
                if fn_rate < best_fnr:
                    best_fnr_tau = tau
                    best_fnr = fn_rate
            print("tau={}, fpr={}, fnr={}".format(best_fnr_tau, fpr(best_fnr_tau), best_fnr))
            return best_fnr_tau, fpr(best_fnr_tau), fnr(best_fnr_tau)


    def train(self, xs, ys, epochs):
        """
        Train the model and setup the two amqs.
        """
        # Filter pos/neg examples
        # TODO: make more efficient (don't necessarily need to compute pos/negs here)
        positives = [x for x,y in zip(xs,ys) if y]
        negatives = [x for x,y in zip(xs,ys) if not y]
        
        # Setup first filter
        self.amq1.add_set(positives)
        
        # Train the neural net on reported positives of first filter
        amq1_pos_indices = [i for i,x in enumerate(xs) if self.amq1.contains(x)]
        amq1_pos_xs = [xs[i] for i in amq1_pos_indices]
        amq1_pos_ys = [ys[i] for i in amq1_pos_indices]

        self.model.train(amq1_pos_xs, amq1_pos_ys, epochs)

        # Tune tau
        self.tau, fpr, fnr = self._choose_tau(amq1_pos_xs, amq1_pos_ys)

        # Get false negatives from model
        model_false_negs = [x for x in amq1_pos_xs
                            if not (self.model(x) > self.tau)]
        num_model_false_negs = len(model_false_negs)
        
        # Setup second filter if we have false negs
        if num_model_false_negs > 0 and fnr > 0:
            # Compute optimal bitarray size ratio for second filter
            inside = fpr/((1-fpr)*(1/fnr - 1))
            m2 = int(0 if inside == 0 else -log2(inside)/log(2))
            if m2 == 0:
                self.amq2 = WordBloom(Bloom.init_ne(num_model_false_negs, self.err))
            else:
                self.amq2 = WordBloom(Bloom.init_nm(num_model_false_negs, m2))
            self.amq2.add_set(model_false_negs)

    def contains(self, x):
        """
        Check if x is in the filter
        """
        # Check the first filter
        # Return output if negative, otherwise continue
        amq1_result = self.amq1.contains(x)
        if not amq1_result:
            return False
        
        # Check model
        # Return output if positive, otherwise continue
        model_result = self.model(x) > self.tau
        if model_result:
            return True

        # Check second filter result, output directly
        return self.amq2.contains(x)

    def __len__(self):
        return (len(self.amq1) + len(self.model) + 
                0 if self.amq2 is None else len(self.amq2))

    def __str__(self):
        return ("[Sandwich] size={}, n={}, c={}, err={}, err1={},"
                "amq1: {}, model: {}, amq2: {}"
                .format(len(self), self.n, self.c, self.err, self.err1,
                        self.amq1, self.model, self.amq2))
