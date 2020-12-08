import torch as T
from bloom import Bloom, WordBloom
from model import WordNet
from math import log

class Sandwich:

    def __init__(self, n, c, err, b_1=4):
        """
        n: number of letters in string
        c: size of alphabet
        err: total error rate of sandwich
        b_1: the amount of space to use for the first bloom filter
        """
        self.n = n
        self.c = c
        self.model = WordNet(n, c)
        self.tau = 0.5 # default value, adjust by tuning later
        self.alpha = 0.618503137801576 # 2 ** -log(2)

        # AMQs can only be set up after training model
        self.amq1 = WordBloom(Bloom.init_ne(n, err*b_1))
        self.amq2 = None # Determine size after training
        self.err = err

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
        # print(candidates)

        # If no tau has fpr < err/2, choose tau with best fpr
        if not candidates:
            print("tau={}, fpr={}, fnr={}".format(best_fpr_tau, best_fpr, fnr(best_fpr_tau)))
            return best_fpr_tau
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
            return best_fnr_tau


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
        amq1_pos_indices = [i for i,x in enumerate(xs) if self.amq.contains(x)]
        amq1_pos_xs = [xs[i] for i in amq1_pos_indices]
        amq1_pos_ys = [ys[i] for i in amq1_pos_indices]

        self.model.train(amq1_pos_xs, amq1_pos_ys, epochs)

        # Tune tau
        self.tau = self._choose_tau(amq1_pos_xs, amq1_pos_ys)

        # Get false negatives from model
        model_false_negs = [x for x in amq1_pos_xs
                            if not (self.model(x) > self.tau)]
        num_model_false_negs = len(model_false_negs)
        
        # Setup second filter if we have false negs
        if num_model_false_negs > 0:
            # Get the examples that were given to the model that were actually negatives
            # and count how many were classified as positive
            

            num_model_false_pos = ilen(x for x,y in zip(amq1_pos_xs, amq1_pos_ys)
                                       if (not y) and (self.model(x) > self.tau))

            # Count the total number of examples given to the model 
            # (i.e, reported positives for amq1)
            num_model_input = len(amq1_pos_ys)
            # Count the number of 1s of exs given to model
            num_pos_in_model_input = sum(amq1_pos_ys)
            # Compute number of 0s of exs given to model
            num_neg_in_model_input = num_model_input - num_pos_in_model_input

            fpr = num_model_false_pos / num_neg_in_model_input
            fnr = num_model_false_negs / num_pos_in_model_input

            # Compute optimal bitarray size ratio for second filter
            m2 = -log2(fpr/((1-fpr)*(1/fnr - 1)))/log(2)
            self.amq2 = WordBloom(Bloom.init_nm(num_model_false_negs, m2))
            self.amq.add_set(model_false_negs)

    def contains(self, x):
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

    def __str__(self):
        return ("n={}, c={}, err={}, b_1={}, amq1: {}, model: {}, amq2: {}"
                .format(self.n, self.c, self.err, self.b_1,
                        self.amq1, self.model, self.amq2))


    def __init__(self, n, c, err, b_1=4):
        """
        n: number of letters in string
        c: size of alphabet
        err: total error rate of sandwich
        b_1: the amount of space to use for the first bloom filter
        """
        self.model = WordNet(n, c)
        self.tau = 0.5 # default value, adjust by tuning later
        self.alpha = 0.618503137801576 # 2 ** -log(2)

        # AMQs can only be set up after training model
        self.amq1 = WordBloom(Bloom.init_ne(n, err*b_1))
        self.amq2 = None # Determine size after training
        self.err = err




        pass
