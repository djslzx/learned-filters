# learned-filters
Experimental work on learning-augmented AMQs

## technical stack
Requires `bitarray`, `mmh3`, `torch`

## Approach
### Filter architectures
- [x] Bloom filter
- [x] Open sandwich bloom filter (Kraska)
- [ ] Sandwiched bloom filter (Mitzenmacher)
- [ ] Model-hash bloom filter (Kraska)

### Basic model
- [x] simple neural net
- [ ] uniformly random test set (details below)

### Extensions
- [ ] other learning algos
- [ ] other test sets (Zipfian: network traces)
- [ ] other architectures

### Testing setup
#### Input type
Bloom filter needs a hashable input, and the model needs matrix of inputs in [0,1].
To meet both requirements, we use the following approach:

Our abstraction is that we are working with strings of `n` letters on an alphabet of size `C`.

The model gets a `1x(Cn)` row vector of values in `{0, 1}` where only one of the scalars in the vector is 1, all others 0.
In other words, for the letter with index `0 <= i < C`, the vector has 0s everywhere except at location `i`, which has value 1.

The filter gets an `n`-element tuple of integers from `0` to `C-1`, which is hashable.

U is the universe (set of n-letter strings from alphabet size C)
K is the set of keys (subset of U)
U - K is the set of negatives

|U| = n ^ C

Can easily convert 'real' numbers by using C=10, converting each number's decimal place to a letter in [0..9]

We handle these conversions by adding a Word type, along with WordNet and WordBloom wrappers for Net and Bloom, respectively.

### Open-faced sandwich (Kraska)
0. Generate training/test sets
   - training set: all positives, some negatives sampled from negative distribution (90%)
   - test set: all positives, some negatives sampled from negative distribution (10%)
1. Train a (nn) model on all positives and some negatives according to assumed distribution
2. Query model using all actual positives, get false negatives
3. Use false negatives to build a filter
4. Connect together
5. Test!

## Estimating model size
Take matrices, count num elts, mult by size of individual elt


## Comparing Filters
- Give each filter the same fixed amount of space, and see how well they do on false positive rate
- Sets:
  - One where the modeling assumption of consistent distribution for negative queries is met,
  - One where this assumption is not met

## Hyperparameters
- heuristic for choosing tau
- training time/epochs
- neural net hyperparams
- ML model


## Lingering questions
- How much do we train the neural net / pre-filter?
- Which negative distributions should we expect to do the best on? Uniform?

## TODO
- [x] Build overflow Bloom filter
- [x] Testing rig (setup sending different inputs to model vs filter)
- [ ] Sandwich
- [ ] Tuning
  - figure out how to get models in sandwiches to perform well enough
    to not make filter worse
  - tuning tau gets fpr under desired threshold
- [ ] saving models and making deterministic inputs (so we can reuse models)
