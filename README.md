# learned-filters
Experimental work on learning-augmented AMQs

## Technical stack
Requires `bitarray`, `mmh3`, and `torch`, all available on pip.

## How to use
To learn how to run tests, go to the `src` directory and run
```
python test.py -h
```
This will display a help message that lists all of the available options.

### Quick start
Run the following to see the bloom filter in action. 
(Use `toast` or `sandwich` in place of `bloom` to see how well the other filters do on the default parameters.)
```
python test.py -f bloom
```


## Approach
### Filter architectures
- [x] Bloom filter
- [x] Open sandwich bloom filter (Kraska)
- [x] Sandwiched bloom filter (Mitzenmacher)

### Testing setup
#### The `Word` type
The Bloom filter needs a hashable input, and the model needs matrix of inputs in [0,1].
To meet both requirements, we use the following approach:

Our abstraction is that we are working with strings of `n` letters on an alphabet of size `C`.

The model gets a `1x(Cn)` row vector of values in `{0, 1}` where only one of the scalars in the vector is 1, all others 0.
In other words, for the letter with index `0 <= i < C`, the vector has 0s everywhere except at location `i`, which has value 1.

The filter gets an `n`-element tuple of integers from `0` to `C-1`, which is hashable.

<!-- U is the universe (set of n-letter strings from alphabet size C) -->
<!-- K is the set of keys (subset of U) -->
<!-- U - K is the set of negatives -->
<!-- |U| = c ^ n -->

Note: We can convert decimal numbers to `Word`s by using C=10, converting each number's decimal place to a letter in [0..9]

We handle these conversions by adding a Word type, along with WordNet and WordBloom wrappers for Net and Bloom, respectively.

## Estimating model size
We estimate the size of the machine learniing model by counting the number of elements in the two matrices that comprise the model's layers, and then multiply by the size of a float in python (24 bytes).

## Hyperparameters
- heuristic for choosing tau
- training time/epochs
- neural net hyperparams
- ML model

## Lingering questions
- How much do we train the neural net / pre-filter?

## TODO
- [x] Build overflow Bloom filter
- [x] Testing rig (setup sending different inputs to model vs filter)
- [x] Sandwich
- [x] Tuning tau
