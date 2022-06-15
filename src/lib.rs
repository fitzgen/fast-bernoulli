#![doc = include_str!("../README.md")]
#![deny(missing_debug_implementations, missing_docs)]
#![forbid(unsafe_code)]

// [What follows is another outstanding comment from Jim Blandy explaining why
// this technique works.]
//
// This comment should just read, "Generate skip counts with a geometric
// distribution", and leave everyone to go look that up and see why it's the
// right thing to do, if they don't know already.
//
// BUT IF YOU'RE CURIOUS, COMMENTS ARE FREE...
//
// Instead of generating a fresh random number for every trial, we can
// randomly generate a count of how many times we should return false before
// the next time we return true. We call this a "skip count". Once we've
// returned true, we generate a fresh skip count, and begin counting down
// again.
//
// Here's an awesome fact: by exercising a little care in the way we generate
// skip counts, we can produce results indistinguishable from those we would
// get "rolling the dice" afresh for every trial.
//
// In short, skip counts in Bernoulli trials of probability `P` obey a geometric
// distribution. If a random variable `X` is uniformly distributed from
// `[0..1)`, then `floor(log(X) / log(1-P))` has the appropriate geometric
// distribution for the skip counts.
//
// Why that formula?
//
// Suppose we're to return `true` with some probability `P`, say, `0.3`. Spread
// all possible futures along a line segment of length `1`. In portion `P` of
// those cases, we'll return true on the next call to `trial`; the skip count is
// 0. For the remaining portion `1-P` of cases, the skip count is `1` or more.
//
// ```
//    skip:             0                         1 or more
//             |------------------^-----------------------------------------|
// portion:            0.3                            0.7
//                      P                             1-P
// ```
//
// But the "1 or more" section of the line is subdivided the same way: *within
// that section*, in portion `P` the second call to `trial()` returns `true`, and
// in portion `1-P` it returns `false` a second time; the skip count is two or
// more. So we return `true` on the second call in proportion `0.7 * 0.3`, and
// skip at least the first two in proportion `0.7 * 0.7`.
//
// ```
//    skip:             0                1              2 or more
//             |------------------^------------^----------------------------|
// portion:            0.3           0.7 * 0.3          0.7 * 0.7
//                      P             (1-P)*P            (1-P)^2
// ```
//
// We can continue to subdivide:
//
// ```
// skip >= 0:  |------------------------------------------------- (1-P)^0 --|
// skip >= 1:  |                  ------------------------------- (1-P)^1 --|
// skip >= 2:  |                               ------------------ (1-P)^2 --|
// skip >= 3:  |                                 ^     ---------- (1-P)^3 --|
// skip >= 4:  |                                 .            --- (1-P)^4 --|
//                                               .
//                                               ^X, see below
// ```
//
// In other words, the likelihood of the next `n` calls to `trial` returning
// `false` is `(1-P)^n`. The longer a run we require, the more the likelihood
// drops. Further calls may return `false` too, but this is the probability
// we'll skip at least `n`.
//
// This is interesting, because we can pick a point along this line segment and
// see which skip count's range it falls within; the point `X` above, for
// example, is within the ">= 2" range, but not within the ">= 3" range, so it
// designates a skip count of `2`. So if we pick points on the line at random
// and use the skip counts they fall under, that will be indistinguishable from
// generating a fresh random number between `0` and `1` for each trial and
// comparing it to `P`.
//
// So to find the skip count for a point `X`, we must ask: To what whole power
// must we raise `1-P` such that we include `X`, but the next power would
// exclude it? This is exactly `floor(log(X) / log(1-P))`.
//
// Our algorithm is then, simply: When constructed, compute an initial skip
// count. Return `false` from `trial` that many times, and then compute a new
// skip count.
//
// For a call to `multi_trial(n)`, if the skip count is greater than `n`, return
// `false` and subtract `n` from the skip count. If the skip count is less than
// `n`, return true and compute a new skip count. Since each trial is
// independent, it doesn't matter by how much `n` overshoots the skip count; we
// can actually compute a new skip count at *any* time without affecting the
// distribution. This is really beautiful.

use rand::Rng;

/// Fast Bernoulli sampling: each event has equal probability of being sampled.
///
/// See the [crate-level documentation][crate] for more general
/// information.
///
/// # Example
///
/// ```
/// use fast_bernoulli::FastBernoulli;
/// use rand::Rng;
///
/// // Get the thread-local random number generator.
/// let mut rng = rand::thread_rng();
///
/// // Create a `FastBernoulli` instance that samples events with probability 1/20.
/// let mut bernoulli = FastBernoulli::new(0.05, &mut rng);
///
/// // Each time your event occurs, perform a Bernoulli trail to determine whether
/// // you should sample the event or not.
/// let on_my_event = || {
///     if bernoulli.trial(&mut rng) {
///         // Record the sample...
///     }
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FastBernoulli {
    probability: f64,
    skip_count: u32,
}

impl FastBernoulli {
    /// Construct a new `FastBernoulli` instance that samples events with the
    /// given probability.
    ///
    /// # Panics
    ///
    /// The probability must be within the range `0.0 <= probability <= 1.0` and
    /// this method will panic if that is not the case.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::Rng;
    /// use fast_bernoulli::FastBernoulli;
    ///
    /// let mut rng = rand::thread_rng();
    /// let sample_one_in_a_hundred = FastBernoulli::new(0.01, &mut rng);
    /// ```
    pub fn new<R>(probability: f64, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        assert!(
            0.0 <= probability && probability <= 1.0,
            "`probability` must be in the range `0.0 <= probability <= 1.0`"
        );

        let mut bernoulli = FastBernoulli {
            probability,
            skip_count: 0,
        };
        bernoulli.reset_skip_count(rng);
        bernoulli
    }

    fn reset_skip_count<R>(&mut self, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        if self.probability == 0.0 {
            // Edge case: we will never sample any event.
            self.skip_count = u32::MAX;
        } else if self.probability == 1.0 {
            // Edge case: we will sample every event.
            self.skip_count = 0;
        } else {
            // Common case: we need to choose a new skip count using the
            // formula `floor(log(x) / log(1 - P))`, as explained in the
            // comment at the top of this file.
            let x: f64 = rng.gen_range(0.0..1.0);
            let skip_count = (x.ln() / (1.0 - self.probability).ln()).floor();
            debug_assert!(skip_count >= 0.0);
            self.skip_count = if skip_count <= (u32::MAX as f64) {
                skip_count as u32
            } else {
                // Clamp the skip count to `u32::MAX`. This can skew
                // sampling when we are sampling with a very low
                // probability, but it is better than any super-robust
                // alternative we have, such as representing skip counts
                // with big nums.
                u32::MAX
            };
        }
    }

    /// Perform a Bernoulli trial: returns `true` with the configured
    /// probability.
    ///
    /// Call this each time an event occurs to determine whether to sample the
    /// event.
    ///
    /// The lower the configured probability, the less overhead calling this
    /// function has.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::Rng;
    /// use fast_bernoulli::FastBernoulli;
    ///
    /// let mut rng = rand::thread_rng();
    /// let mut bernoulli = FastBernoulli::new(0.1, &mut rng);
    ///
    /// // Each time an event occurs, call `trial`...
    /// if bernoulli.trial(&mut rng) {
    ///     // ...and if it returns true, record a sample of this event.
    /// }
    /// ```
    pub fn trial<R>(&mut self, rng: &mut R) -> bool
    where
        R: Rng + ?Sized,
    {
        if self.skip_count > 0 {
            self.skip_count -= 1;
            return false;
        }

        self.reset_skip_count(rng);
        self.probability != 0.0
    }

    /// Perform `n` Bernoulli trials at once.
    ///
    /// This is semantically equivalent to calling the `trial()` method `n`
    /// times and returning `true` if any of those calls returned `true`, but
    /// runs in `O(1)` time instead of `O(n)` time.
    ///
    /// What is this good for? In some applications, some events are "bigger"
    /// than others. For example, large memory allocations are more significant
    /// than small memory allocations. Perhaps we'd like to imagine that we're
    /// drawing allocations from a stream of bytes, and performing a separate
    /// Bernoulli trial on every byte from the stream. We can accomplish this by
    /// calling `multi_trial(s)` for the number of bytes `s`, and sampling the
    /// event if that call returns true.
    ///
    /// Of course, this style of sampling needs to be paired with analysis and
    /// presentation that makes the "size" of the event apparent, lest trials
    /// with large values for `n` appear to be indistinguishable from those with
    /// small values for `n`, despite being potentially much more likely to be
    /// sampled.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::Rng;
    /// use fast_bernoulli::FastBernoulli;
    ///
    /// let mut rng = rand::thread_rng();
    /// let mut byte_sampler = FastBernoulli::new(0.05, &mut rng);
    ///
    /// // When we observe a `malloc` of ten bytes event...
    /// if byte_sampler.multi_trial(10, &mut rng) {
    ///     // ... if `multi_trial` returns `true` then we sample it.
    ///     record_malloc_sample(10);
    /// }
    ///
    /// // And when we observe a `malloc` of 1024 bytes event...
    /// if byte_sampler.multi_trial(1024, &mut rng) {
    ///     // ... if `multi_trial` returns `true` then we sample this larger
    ///     // allocation.
    ///     record_malloc_sample(1024);
    /// }
    /// # fn record_malloc_sample(_: u32) {}
    /// ```
    pub fn multi_trial<R>(&mut self, n: u32, rng: &mut R) -> bool
    where
        R: Rng + ?Sized,
    {
        if n < self.skip_count {
            self.skip_count -= n;
            return false;
        }

        self.reset_skip_count(rng);
        self.probability != 0.0
    }

    /// Get the probability with which events are sampled.
    ///
    /// This is a number between `0.0` and `1.0`.
    ///
    /// This is the same value that was passed to `FastBernoulli::new` when
    /// constructing this instance.
    #[inline]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// How many events will be skipped until the next event is sampled?
    ///
    /// When `self.probability() == 0.0` this method's return value is
    /// inaccurate, and logically should be infinity.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::Rng;
    /// use fast_bernoulli::FastBernoulli;
    ///
    /// let mut rng = rand::thread_rng();
    /// let mut bernoulli = FastBernoulli::new(0.1, &mut rng);
    ///
    /// // Get the number of upcoming events that will not be sampled.
    /// let skip_count = bernoulli.skip_count();
    ///
    /// // That many events will not be sampled.
    /// for _ in 0..skip_count {
    ///     assert!(!bernoulli.trial(&mut rng));
    /// }
    ///
    /// // The next event will be sampled.
    /// assert!(bernoulli.trial(&mut rng));
    /// ```
    #[inline]
    pub fn skip_count(&self) -> u32 {
        self.skip_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_number_of_samples() {
        let mut rng = rand::thread_rng();

        let probability = 0.01;
        let events = 10_000;
        let expected = (events as f64) * probability;
        let error_tolerance = expected * 0.25;

        let mut bernoulli = FastBernoulli::new(probability, &mut rng);

        let mut num_sampled = 0;
        for _ in 0..events {
            if bernoulli.trial(&mut rng) {
                num_sampled += 1;
            }
        }

        let min = (expected - error_tolerance) as u32;
        let max = (expected + error_tolerance) as u32;
        assert!(
            min <= num_sampled && num_sampled <= max,
            "expected ~{} samples, found {} (acceptable range is {} to {})",
            expected,
            num_sampled,
            min,
            max,
        );
    }
}
