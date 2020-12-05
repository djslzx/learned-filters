# learned-filters
Experimental work on learning-augmented AMQs

## technical stack
Requires `bitarray`, `mmh3`, `torch`

## Approach
### Filter architectures
- [x] Bloom filter
- [ ] Overflow bloom filter (Kraska)
- [ ] Sandwiched bloom filter (Mitzenmacher)
- [ ] Model-hash bloom filter (Kraska)

### Basic model
- [ ] simple neural net
- [ ] uniformly random test set

### Extensions
- [ ] other learning algos
- [ ] other test sets (Zipfian: network traces)
- [ ] other architectures
