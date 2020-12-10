import torch as T
from bloom import WordBloom, Bloom
from model import WordNet
from util import ilen

class Toast:

    def __init__(self, n, c, err):
        """
        n: number of letters in string
        c: size of alphabet
        """
        self.model = WordNet(n, c)
        self.tau = 0.5

        # AMQ can only be set up after training model
        self.amq = None
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
        Train on examples for a certain number of epochs
        """
        # Train neural net
        # Note: torch dataloader takes care of shuffling
        self.model.train(xs, ys, epochs)

        # Tune tau
        self.tau = self._choose_tau(xs, ys)

        # Get false negatives
        positives = [x for x,y in zip(xs,ys) if y]
        false_negs = [x for x in positives
                      if not (self.model(x) > self.tau)]
        
        # Build filter for negatives
        if len(false_negs) > 0:
            self.amq = WordBloom(Bloom.init_ne(len(false_negs), self.err/2))
            self.amq.add_set(false_negs)

    def contains(self, x):
        """
        Check if x is in the filter
        """
        # Check amq only if model reports negative and model has false negatives
        model_ans = self.model(x) > self.tau
        if model_ans or self.amq is None:
            return model_ans
        else:
            return self.amq.contains(x)

    def __len__(self):
        return len(self.model) + (0 if self.amq is None else len(self.amq))

    def __str__(self):
        return ("[Toast] size={}, amq: {}, model: {}, e={}, tau={}"
                .format(len(self), self.amq, self.model, self.err, self.tau))
