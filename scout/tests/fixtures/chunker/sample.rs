//! Fixture: top-level items of a Rust module.
use std::collections::HashMap;

/// Word frequencies gathered from one or more text inputs, keyed by lowercase word.
pub struct Tally {
    counts: HashMap<String, usize>,
    total: usize,
}

impl Tally {
    pub fn new() -> Self {
        Tally { counts: HashMap::new(), total: 0 }
    }

    pub fn add(&mut self, text: &str) {
        for word in text.split_whitespace() {
            *self.counts.entry(word.to_lowercase()).or_insert(0) += 1;
            self.total += 1;
        }
    }
}

/// Return the most frequent word, breaking ties by the alphabetically smaller one.
pub fn top_word(tally: &Tally) -> Option<&str> {
    tally
        .counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(word, _)| word.as_str())
}
