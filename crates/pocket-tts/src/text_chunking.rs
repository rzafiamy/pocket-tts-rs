//! Turning a text prompt into the chunks the model generates one at a time.
//!
//! Port of Python's `pocket_tts/models/text_chunking.py`. The model is trained
//! on single sentences, so long inputs are split on sentence boundaries and
//! regrouped into chunks that fit `max_tokens`.

use crate::conditioners::text::LUTConditioner;
use crate::config::Config;
use anyhow::Result;
use std::collections::{BTreeMap, HashSet};

const TERMINAL_PUNCTUATION: &[char] = &['.', '!', '?', '\u{2026}'];
const WEAK_PUNCTUATION: &[char] = &[',', ';', ':', '-', '\u{2013}', '\u{2014}'];
const CLOSERS: &[char] = &['"', '\'', '\u{201d}', '\u{2019}', ')', ']', '\u{00bb}'];

/// Text preparation options, from the model config.
#[derive(Debug, Clone, Default)]
pub struct TextOptions {
    pub pad_with_spaces_for_short_inputs: bool,
    pub remove_semicolons: bool,
    pub append_terminal_punctuation: bool,
    pub capitalize_first_letter: bool,
    pub replace_characters: BTreeMap<String, String>,
}

impl From<&Config> for TextOptions {
    fn from(c: &Config) -> Self {
        Self {
            pad_with_spaces_for_short_inputs: c.pad_with_spaces_for_short_inputs,
            remove_semicolons: c.remove_semicolons,
            append_terminal_punctuation: c.append_terminal_punctuation,
            capitalize_first_letter: c.capitalize_first_letter,
            replace_characters: c.replace_characters.clone(),
        }
    }
}

/// Normalizes one prompt; returns it with the guessed number of frames to keep
/// generating after EOS.
pub fn prepare_text_prompt(text: &str, opts: &TextOptions) -> Result<(String, usize)> {
    let mut text = text.trim().to_string();
    if !opts.replace_characters.is_empty() {
        let translated: String = text
            .chars()
            .map(|c| {
                let mut buf = [0u8; 4];
                match opts.replace_characters.get(&*c.encode_utf8(&mut buf)) {
                    Some(r) => r.clone(),
                    None => c.to_string(),
                }
            })
            .collect();
        text = translated.split_whitespace().collect::<Vec<_>>().join(" ");
        // Deleted quotes leave '"Hi?", she said' as 'Hi?, she said', which reads as a
        // sentence end followed by a stray comma; keep the sentence mark only.
        text = drop_weak_after_terminal(&text);
    }
    if text.is_empty() {
        anyhow::bail!("Text prompt cannot be empty");
    }
    text = text.replace(['\n', '\r'], " ").replace("  ", " ");
    if opts.remove_semicolons {
        text = text.replace(';', ",");
    }
    let frames_after_eos_guess = if text.split_whitespace().count() <= 4 {
        3
    } else {
        1
    };

    if opts.capitalize_first_letter {
        let first = text.chars().next().expect("non-empty");
        if !first.is_uppercase() {
            text = format!("{}{}", first.to_uppercase(), &text[first.len_utf8()..]);
        }
    }

    if opts.append_terminal_punctuation {
        text = ensure_terminal_punctuation(&text);
    }

    // The model does not perform well with very few tokens; leading spaces
    // increase the token count.
    if opts.pad_with_spaces_for_short_inputs && text.split_whitespace().count() < 5 {
        text = format!("{}{}", " ".repeat(8), text);
    }

    Ok((text, frames_after_eos_guess))
}

/// `re.sub(r"([.!?…])\s*[,;:]", r"\1", text)`
fn drop_weak_after_terminal(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        out.push(c);
        i += 1;
        if TERMINAL_PUNCTUATION.contains(&c) {
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && matches!(chars[j], ',' | ';' | ':') {
                i = j + 1;
            }
        }
    }
    out
}

/// Makes sure the prompt ends with sentence-final punctuation, after any
/// closing quotes or brackets. A trailing comma, colon or dash becomes a period.
fn ensure_terminal_punctuation(text: &str) -> String {
    let core = text.trim_end_matches(|c: char| CLOSERS.contains(&c) || c == ' ');
    let closers = text[core.len()..].trim();
    match core.chars().last() {
        None => text.to_string(),
        Some(c) if TERMINAL_PUNCTUATION.contains(&c) => text.to_string(),
        Some(c) if WEAK_PUNCTUATION.contains(&c) => {
            let stripped =
                core.trim_end_matches(|c: char| WEAK_PUNCTUATION.contains(&c) || c == ' ');
            format!("{stripped}.{closers}")
        }
        Some(_) => format!("{text}."),
    }
}

/// True when `start` begins right after a decimal period ("3.|14").
fn is_decimal_period_boundary(
    tokens: &[u32],
    start: usize,
    tokenizer: &LUTConditioner,
) -> Result<bool> {
    let prefix: Vec<char> = tokenizer.decode(&tokens[..start])?.chars().collect();
    let suffix = tokenizer.decode(&tokens[start..])?;
    let n = prefix.len();
    Ok(n >= 2
        && prefix[n - 1] == '.'
        && prefix[n - 2].is_ascii_digit()
        && suffix.chars().next().is_some_and(|c| c.is_ascii_digit()))
}

/// Token indices delimiting segments: each segment ends after a run of
/// boundary tokens. Starts with 0 and ends with `tokens.len()`.
fn find_boundary_indices(
    tokens: &[u32],
    boundary_tokens: &[u32],
    decimal_check: Option<&LUTConditioner>,
) -> Result<Vec<usize>> {
    let boundary: HashSet<u32> = boundary_tokens.iter().copied().collect();
    let mut indices = vec![0];
    let mut previous_was_boundary = false;
    for (idx, token) in tokens.iter().enumerate() {
        if boundary.contains(token) {
            previous_was_boundary = true;
            continue;
        }
        if previous_was_boundary {
            if let Some(tok) = decimal_check
                && is_decimal_period_boundary(tokens, idx, tok)?
            {
                previous_was_boundary = false;
                continue;
            }
            indices.push(idx);
        }
        previous_was_boundary = false;
    }
    indices.push(tokens.len());
    Ok(indices)
}

fn segments_from_boundaries(
    tokens: &[u32],
    boundaries: &[usize],
    tokenizer: &LUTConditioner,
) -> Result<Vec<(usize, String)>> {
    boundaries
        .windows(2)
        .map(|w| Ok((w[1] - w[0], tokenizer.decode(&tokens[w[0]..w[1]])?)))
        .collect()
}

/// Splits `text` into chunks of at most `max_tokens` tokens on sentence
/// boundaries, sub-splitting oversized sentences on commas, semicolons and colons.
pub fn split_into_best_sentences(
    tokenizer: &LUTConditioner,
    text: &str,
    max_tokens: usize,
    opts: &TextOptions,
) -> Result<Vec<String>> {
    let (text, _) = prepare_text_prompt(text, opts)?;
    let text = text.trim();
    let tokens = tokenizer.encode(text)?;

    // The first id of each probe is the word-start marker, not punctuation.
    let end_of_sentence: Vec<u32> = tokenizer.encode(".!...?")?.into_iter().skip(1).collect();
    let boundaries = find_boundary_indices(&tokens, &end_of_sentence, Some(tokenizer))?;
    let sentences = segments_from_boundaries(&tokens, &boundaries, tokenizer)?;

    let fallback: Vec<u32> = tokenizer.encode(",;:")?.into_iter().skip(1).collect();
    let mut refined = Vec::new();
    for (nb_tokens, sentence) in sentences {
        if nb_tokens <= max_tokens {
            refined.push((nb_tokens, sentence));
            continue;
        }
        let sub_tokens = tokenizer.encode(sentence.trim())?;
        let sub_boundaries = find_boundary_indices(&sub_tokens, &fallback, None)?;
        let sub_segments = segments_from_boundaries(&sub_tokens, &sub_boundaries, tokenizer)?;
        if sub_segments.len() > 1 {
            refined.extend(sub_segments);
        } else {
            refined.push((nb_tokens, sentence));
        }
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0;
    for (nb_tokens, sentence) in refined {
        if current.is_empty() {
            current = sentence;
            current_tokens = nb_tokens;
        } else if current_tokens + nb_tokens > max_tokens {
            chunks.push(current.trim().to_string());
            current = sentence;
            current_tokens = nb_tokens;
        } else {
            current.push(' ');
            current.push_str(&sentence);
            current_tokens += nb_tokens;
        }
    }
    if !current.is_empty() {
        chunks.push(current.trim().to_string());
    }

    for chunk in &chunks {
        let n = tokenizer.encode(chunk.trim())?.len();
        if n > max_tokens {
            tracing::warn!(
                "Chunk has {n} tokens (max {max_tokens}), generation may skip words: '{}...'",
                chunk.chars().take(50).collect::<String>()
            );
        }
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> TextOptions {
        TextOptions {
            append_terminal_punctuation: true,
            capitalize_first_letter: true,
            ..Default::default()
        }
    }

    /// covers: REQ-TXT-001
    #[test]
    fn terminal_punctuation() {
        assert_eq!(ensure_terminal_punctuation("Hello"), "Hello.");
        assert_eq!(ensure_terminal_punctuation("Hello!"), "Hello!");
        assert_eq!(ensure_terminal_punctuation("Hello,"), "Hello.");
        assert_eq!(ensure_terminal_punctuation("\"Hello\""), "\"Hello\".");
        assert_eq!(ensure_terminal_punctuation("\"Hello.\""), "\"Hello.\"");
        assert_eq!(ensure_terminal_punctuation("Hello -"), "Hello.");
        assert_eq!(ensure_terminal_punctuation("(wait, ) "), "(wait.)");
    }

    /// covers: REQ-TXT-001
    #[test]
    fn prepare_basic() {
        let (t, g) = prepare_text_prompt("  hello world ", &opts()).unwrap();
        assert_eq!((t.as_str(), g), ("Hello world.", 3));
        let (t, g) = prepare_text_prompt("one two three four five", &opts()).unwrap();
        assert_eq!((t.as_str(), g), ("One two three four five.", 1));
        assert!(prepare_text_prompt("   ", &opts()).is_err());
    }

    /// covers: REQ-TXT-001
    #[test]
    fn prepare_padding_and_semicolons() {
        let o = TextOptions {
            pad_with_spaces_for_short_inputs: true,
            remove_semicolons: true,
            ..opts()
        };
        let (t, _) = prepare_text_prompt("a; b", &o).unwrap();
        assert_eq!(t, "        A, b.");
    }

    /// covers: REQ-TXT-001
    #[test]
    fn prepare_replace_characters() {
        let mut o = opts();
        o.replace_characters.insert("\"".into(), "".into());
        o.replace_characters.insert("\u{2019}".into(), "'".into());
        let (t, _) = prepare_text_prompt("\"Hi?\", she said l\u{2019}air", &o).unwrap();
        assert_eq!(t, "Hi? she said l'air.");
    }
}
