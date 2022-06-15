# `fast-bernoulli`: Efficient sampling with uniform probability

[![Rust](https://github.com/fitzgen/fast-bernoulli/actions/workflows/rust.yml/badge.svg)](https://github.com/fitzgen/fast-bernoulli/actions/workflows/rust.yml)

When gathering statistics about a program's behavior, we may be observing events
that occur very frequently (e.g., function calls or memory allocations) and we
may be gathering information that is somewhat expensive to produce (e.g., call
stacks). Sampling all the events could have a significant impact on the
program's performance.

Why not just sample every `N`'th event? This technique is called "systematic
sampling"; it's simple and efficient, and it's fine if we imagine a patternless
stream of events. But what if we're sampling allocations, and the program
happens to have a loop where each iteration does exactly `N` allocations? You
would end up sampling the same allocation every time through the loop; the
entire rest of the loop becomes invisible to your measurements! More generally,
if each iteration does `M` allocations, and `M` and `N` have any common divisor
at all, most allocation sites will never be sampled. If they're both even, say,
the odd-numbered allocations disappear from your results.

Ideally, we'd like each event to have some probability `P` of being sampled,
independent of its neighbors and of its position in the sequence. This is called
["Bernoulli sampling"][bernoulli-sampling], and it doesn't suffer from any of
the problems mentioned above.

[bernoulli-sampling]: https://en.wikipedia.org/wiki/Bernoulli_sampling

One disadvantage of Bernoulli sampling is that you can't be sure exactly how
many samples you'll get: technically, it's possible that you might sample none
of them, or all of them. But if the number of events `N` is large, these aren't
likely outcomes; you can generally expect somewhere around `P * N` events to be
sampled.

The other disadvantage of Bernoulli sampling is that you have to generate a
random number for every event, which can be slow.

> &lt;significant pause&gt;

BUT NOT WITH THIS CRATE!

`FastBernoulli` lets you do true Bernoulli sampling, while generating a fresh
random number only when we do decide to sample an event, not on every
trial. When it decides not to sample, a call to `FastBernoulli::trial` is
nothing but decrementing a counter and comparing it to zero. So the lower your
sampling probability is, the less overhead `FastBernoulli` imposes.

Finally, probabilities of `0` and `1` are handled efficiently. (In neither case
need we ever generate a random number at all.)

## Example

```rust
use fast_bernoulli::FastBernoulli;
use rand::Rng;

// Get the thread-local random number generator.
let mut rng = rand::thread_rng();

// Create a `FastBernoulli` instance that samples events with probability 1/20.
let mut bernoulli = FastBernoulli::new(0.05, &mut rng);

// Each time your event occurs, perform a Bernoulli trail to determine whether
// you should sample the event or not.
let on_my_event = || {
    if bernoulli.trial(&mut rng) {
        // Record the sample...
    }
};
```

## Inspiration

This crate uses the same technique that [Jim Blandy] used for [the
`FastBernoulliTrial` class][firefox-class] in Firefox. This implementation is
not a direct transcription of that C++ to Rust, however I did copy (and lightly
edit) some documentation and comments from the original (for example, most of
this README / crate-level documentation).

[Jim Blandy]: https://www.red-bean.com/~jimb/
[firefox-class]: https://searchfox.org/mozilla-central/rev/a6d25de0c706dbc072407ed5d339aaed1cab43b7/mfbt/FastBernoulliTrial.h
