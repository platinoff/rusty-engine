//! HF `GPT-2` / `qwen2` BPE pre-tokenizer splitter — band B3 `PH-305b` parity.
//!
//! Reference regex (leftmost-first alternation):
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)
//! | [^\r\n\p{L}\p{N}]?\p{L}+
//! | \p{N}{1,3}
//! |  ?[^\s\p{L}\p{N}]+[\r\n]*
//! | \s*[\r\n]+
//! | \s+(?!\S)
//! | \s+
//! ```
//!
//! Implemented with `std` only (no `regex`/dep): `is_alphabetic`≈`\p{L}`,
//! `is_numeric`≈`\p{N}`, `is_whitespace`≈`\s`. Qwen2 reuses the GPT-2 pattern
//! **except the digits alternative**: HF `qwen2` `tokenizer.json` pins `\p{N}`
//! (one digit per piece), not `\p{N}{1,3}` — see [`qwen2_pre_split`].
//! Byte-encoding (`ByteEncoder`) is applied by the caller **after** this split, so
//! a leading single space attaches to the following word — the whole point of the
//! pattern. The matcher backtracks across alternatives exactly as the regex engine
//! would: for a given start position we try each alternative in order and take the
//! first that yields a non-empty match.

/// True for a `\r`/`\n` newline character.
fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// True for a `[^\r\n\p{L}\p{N}]` character (not newline, not letter, not number).
fn is_pseudo_other(c: char) -> bool {
    !is_nl(c) && !c.is_alphabetic() && !c.is_numeric()
}

/// True for a `[^\s\p{L}\p{N}]` character (not whitespace, not letter, not number).
fn is_punct(c: char) -> bool {
    !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric()
}

/// The lowercase contraction tails the first alternative matches after `'`.
const CONTRACTIONS: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];

/// Alternative 1 length at `i` (with apostrophe) if a contraction tail follows, else 0.
fn contraction(cs: &[char], i: usize) -> usize {
    if cs[i] != '\'' {
        return 0;
    }
    let mut low = String::new();
    let mut j = i + 1;
    while j < cs.len() && low.len() < 2 {
        let c = cs[j];
        if c.is_alphabetic() {
            low.extend(c.to_lowercase());
            j += 1;
        } else {
            break;
        }
    }
    let mut best = 0usize;
    for k in CONTRACTIONS {
        if k.len() <= low.len() && low[..k.len()] == *k && k.len() > best {
            best = k.len();
        }
    }
    if best == 0 {
        0
    } else {
        best + 1
    }
}

/// Alternative 2 length at `i`: `[^\r\n\p{L}\p{N}]?\p{L}+` (optional one prefix char,
/// then one or more letters). Returns the match length, or 0 if no letters follow.
fn letters_with_optional_prefix(cs: &[char], i: usize) -> usize {
    let n = cs.len();
    // case a: starts directly on a letter
    if i < n && cs[i].is_alphabetic() {
        let mut j = i;
        while j < n && cs[j].is_alphabetic() {
            j += 1;
        }
        return j - i;
    }
    // case b: optional single [^\r\n LN] prefix, then a letter
    if i < n && is_pseudo_other(cs[i]) {
        let j = i + 1;
        if j < n && cs[j].is_alphabetic() {
            let mut k = j;
            while k < n && cs[k].is_alphabetic() {
                k += 1;
            }
            return k - i;
        }
    }
    0
}

/// Split `text` into the pieces the **GPT-2** pre-tokenizer would hand to BPE
/// (digits grouped in runs of up to three: `\p{N}{1,3}`).
pub fn gpt2_pre_split(text: &str) -> Vec<String> {
    pre_split(text, 3)
}

/// Split `text` into the pieces the HF **qwen2** pre-tokenizer would hand to
/// BPE: same alternation as GPT-2 but `\p{N}` — every digit is its own piece.
pub fn qwen2_pre_split(text: &str) -> Vec<String> {
    pre_split(text, 1)
}

/// Shared splitter; `digit_max` is the `\p{N}` repetition cap (3 = GPT-2, 1 = Qwen2).
fn pre_split(text: &str, digit_max: usize) -> Vec<String> {
    let cs: Vec<char> = text.chars().collect();
    let n = cs.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        // A: contraction ('s|'t|'re|'ve|'m|'ll|'d, case-insensitive tail)
        let len = contraction(&cs, i);
        if len > 0 {
            out.push(cs[i..i + len].iter().collect());
            i += len;
            continue;
        }
        // B: [^\r\n LN]?L+  (this is what attaches a single leading space to a word)
        let len = letters_with_optional_prefix(&cs, i);
        if len > 0 {
            out.push(cs[i..i + len].iter().collect());
            i += len;
            continue;
        }
        // C: N{1,digit_max}
        if cs[i].is_numeric() {
            let end = (i + digit_max).min(n);
            let mut j = i;
            while j < end && cs[j].is_numeric() {
                j += 1;
            }
            out.push(cs[i..j].iter().collect());
            i = j;
            continue;
        }
        // D:  ?[^ LN]+[\r\n]*
        {
            let mut j = i;
            if j < n && cs[j] == ' ' {
                j += 1;
            }
            if j < n && is_punct(cs[j]) {
                while j < n && is_punct(cs[j]) {
                    j += 1;
                }
                while j < n && is_nl(cs[j]) {
                    j += 1;
                }
                out.push(cs[i..j].iter().collect());
                i = j;
                continue;
            }
        }
        // E / F / G: whitespace handling
        if cs[i].is_whitespace() {
            let mut run = i;
            while run < n && cs[run].is_whitespace() {
                run += 1;
            }
            // E: \s*[\r\n]+  → a run whose final char is a newline
            if is_nl(cs[run - 1]) {
                out.push(cs[i..run].iter().collect());
                i = run;
                continue;
            }
            // F: \s+(?!\S)  → trailing whitespace at end-of-string
            if run == n {
                out.push(cs[i..run].iter().collect());
                i = run;
                continue;
            }
            // G: \s+  → any other run is its own piece
            out.push(cs[i..run].iter().collect());
            i = run;
            continue;
        }
        // Fallback: single char (covers a lone `'` that was not a contraction tail).
        out.push(cs[i..i + 1].iter().collect());
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_world_matches_gpt2() {
        assert_eq!(
            gpt2_pre_split("Hello, world!"),
            vec!["Hello", ",", " world", "!"]
        );
    }

    #[test]
    fn contractions_split_out() {
        assert_eq!(gpt2_pre_split("don't"), vec!["don", "'t"]);
        assert_eq!(gpt2_pre_split("it's"), vec!["it", "'s"]);
        assert_eq!(gpt2_pre_split("we'll"), vec!["we", "'ll"]);
    }

    #[test]
    fn digits_grouped_in_threes() {
        assert_eq!(gpt2_pre_split("1234567"), vec!["123", "456", "7"]);
        assert_eq!(gpt2_pre_split("a12b"), vec!["a", "12", "b"]);
    }

    #[test]
    fn qwen2_splits_each_digit_alone() {
        // HF qwen2 tokenizer.json pins the digits alternative to \p{N} (no {1,3}).
        assert_eq!(
            qwen2_pre_split("1234567"),
            vec!["1", "2", "3", "4", "5", "6", "7"]
        );
        assert_eq!(qwen2_pre_split("a12b"), vec!["a", "1", "2", "b"]);
        assert_eq!(
            qwen2_pre_split("2+2="),
            vec!["2", "+", "2", "="],
            "digit runs never join across punctuation in either split"
        );
    }

    #[test]
    fn leading_space_attaches_to_word() {
        assert_eq!(gpt2_pre_split(" hi"), vec![" hi"]);
    }

    #[test]
    fn newline_groups_are_their_own_piece() {
        assert_eq!(gpt2_pre_split("a\n\nb"), vec!["a", "\n\n", "b"]);
    }

    #[test]
    fn symbol_runs_collapse() {
        assert_eq!(gpt2_pre_split("!!"), vec!["!!"]);
    }
}
