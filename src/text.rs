//! Text side of a call: cutting a streamed reply into
//! the pieces that are spoken one at a time, and making each piece sayable.
//!
//! Text-to-speech engines synthesize a whole input before returning any of
//! it, so the first audio a caller hears waits for the first piece to be
//! complete. The chunker therefore cuts as early as it can without cutting
//! badly: at the end of each sentence, and — for the very first piece only —
//! at the first clause boundary once a few words have arrived, so a long
//! opening sentence does not hold the whole reply silent.

/// Words the first piece needs before it may be cut at a comma rather than at
/// a full stop. Fewer would speak "So," on its own; more delays first audio.
const FIRST_CLAUSE_MIN_WORDS: usize = 4;

/// A run of text this long with no boundary is cut at its last space — a
/// model that writes a paragraph-long sentence still gets spoken.
const MAX_PIECE_CHARS: usize = 240;

/// Abbreviations whose full stop does not end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "st.", "no.", "approx.", "cf.",
];

/// Cuts a reply into speakable pieces as its text streams in.
#[derive(Default)]
pub struct SentenceChunker {
    buffer: String,
    emitted: usize,
}

impl SentenceChunker {
    /// An empty chunker, for one reply.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add streamed text; returns every piece now complete, in order.
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.buffer.push_str(delta);
        let mut pieces = Vec::new();
        while let Some(cut) = self.next_cut() {
            let piece: String = self.buffer.drain(..cut).collect();
            if let Some(piece) = self.take(piece) {
                pieces.push(piece);
            }
        }
        pieces
    }

    /// The reply is complete: whatever is left is the last piece.
    pub fn finish(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.buffer);
        self.take(rest)
    }

    fn take(&mut self, piece: String) -> Option<String> {
        let piece = piece.trim().to_string();
        if piece.chars().any(char::is_alphanumeric) {
            self.emitted += 1;
            Some(piece)
        } else {
            None
        }
    }

    /// Byte index just past the next boundary, if the buffer holds one.
    fn next_cut(&self) -> Option<usize> {
        let text = self.buffer.as_str();
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        for (i, &(at, c)) in chars.iter().enumerate() {
            // A boundary is only certain once the character after it has
            // arrived: "3." may be "3.5", and "e.g" may be "e.g.".
            let Some(&(next_at, next)) = chars.get(i + 1) else {
                break;
            };
            if c == '\n' {
                return Some(next_at);
            }
            let end = after_closers(&chars, i + 1);
            let ends_sentence = matches!(c, '.' | '!' | '?' | '…')
                && chars.get(end).is_some_and(|(_, ch)| ch.is_whitespace());
            if ends_sentence && !(c == '.' && is_abbreviation(&text[..=at])) {
                return Some(chars.get(end).map_or(text.len(), |(at, _)| *at));
            }
            let clause = matches!(c, ',' | ';' | ':' | '—') && next.is_whitespace();
            if clause
                && self.emitted == 0
                && text[..at].split_whitespace().count() >= FIRST_CLAUSE_MIN_WORDS
            {
                return Some(next_at);
            }
        }
        if text.len() > MAX_PIECE_CHARS {
            let mut limit = MAX_PIECE_CHARS;
            while !text.is_char_boundary(limit) {
                limit -= 1;
            }
            return text[..limit].rfind(char::is_whitespace).map(|at| at + 1);
        }
        None
    }
}

/// Index of the first character after any closing quotes or brackets that
/// follow a sentence's final punctuation (`He said "stop."` ends after `"`).
fn after_closers(chars: &[(usize, char)], mut i: usize) -> usize {
    while chars
        .get(i)
        .is_some_and(|(_, c)| matches!(c, '"' | '\'' | ')' | ']' | '”' | '’'))
    {
        i += 1;
    }
    i
}

fn is_abbreviation(upto_dot: &str) -> bool {
    let word = upto_dot
        .rsplit(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    // A lone capital letter ("J. Smith") is an initial, not a sentence end.
    let initial = word.len() == 2 && word.starts_with(|c: char| c.is_ascii_alphabetic());
    initial || ABBREVIATIONS.contains(&word.as_str())
}

/// The piece as it should be said: markdown a reader would see past, a
/// speaker would read aloud. Emphasis and code marks go, a link becomes its
/// text, a list bullet or heading mark goes, and a bare URL becomes "a link".
pub fn speakable(piece: &str) -> String {
    let mut out = String::with_capacity(piece.len());
    let mut rest = piece.trim();
    // Leading list bullets, numbering and heading marks.
    loop {
        let trimmed = rest
            .trim_start_matches('#')
            .trim_start_matches(['-', '*', '•', '>'])
            .trim_start();
        let trimmed = strip_list_number(trimmed);
        if trimmed.len() == rest.len() {
            break;
        }
        rest = trimmed;
    }
    let mut chars = rest.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        match c {
            '*' | '`' | '_' if is_markup(rest, at, c) => {}
            '[' => {
                // [text](url) → text
                if let Some(close) = rest[at..].find("](") {
                    let text_end = at + close;
                    if let Some(paren) = rest[text_end + 2..].find(')') {
                        out.push_str(&rest[at + 1..text_end]);
                        let skip_to = text_end + 2 + paren + 1;
                        while chars.peek().is_some_and(|(i, _)| *i < skip_to) {
                            chars.next();
                        }
                        continue;
                    }
                }
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    let words: Vec<String> = out
        .split_whitespace()
        .map(|word| {
            if word.starts_with("http://") || word.starts_with("https://") {
                "a link".to_string()
            } else {
                word.to_string()
            }
        })
        .collect();
    words.join(" ")
}

fn strip_list_number(text: &str) -> &str {
    let digits = text.chars().take_while(char::is_ascii_digit).count();
    if digits > 0 && digits < 3 {
        let after = &text[digits..];
        if let Some(rest) = after
            .strip_prefix(". ")
            .or_else(|| after.strip_prefix(") "))
        {
            return rest.trim_start();
        }
    }
    text
}

/// `_` inside a word (`snake_case`) is part of it; elsewhere, like `*` and
/// `` ` ``, it is emphasis or code markup.
fn is_markup(text: &str, at: usize, c: char) -> bool {
    if c != '_' {
        return true;
    }
    let before = text[..at].chars().next_back();
    let after = text[at + 1..].chars().next();
    !(before.is_some_and(char::is_alphanumeric) && after.is_some_and(char::is_alphanumeric))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stream `text` a few characters at a time, as a model would.
    fn chunk(text: &str, step: usize) -> Vec<String> {
        let mut chunker = SentenceChunker::new();
        let chars: Vec<char> = text.chars().collect();
        let mut pieces = Vec::new();
        for part in chars.chunks(step) {
            pieces.extend(chunker.push(&part.iter().collect::<String>()));
        }
        pieces.extend(chunker.finish());
        pieces
    }

    #[test]
    fn sentences_are_cut_as_they_complete_whatever_the_delta_size() {
        let text = "Two sessions are idle. One is waiting for input! Want me to look?";
        for step in [1, 3, 7, 100] {
            assert_eq!(
                chunk(text, step),
                vec![
                    "Two sessions are idle.",
                    "One is waiting for input!",
                    "Want me to look?"
                ],
                "step {step}"
            );
        }
    }

    #[test]
    fn a_long_first_sentence_is_cut_at_its_first_clause() {
        let text = "The build on main failed overnight, because the runner ran out of disk, \
                    and nothing has retried it. Shall I?";
        assert_eq!(
            chunk(text, 2),
            vec![
                "The build on main failed overnight,",
                "because the runner ran out of disk, and nothing has retried it.",
                "Shall I?"
            ]
        );
    }

    #[test]
    fn a_short_opening_is_not_cut_at_a_comma() {
        assert_eq!(chunk("Yes, it is.", 1), vec!["Yes, it is."]);
    }

    #[test]
    fn numbers_abbreviations_and_initials_do_not_end_a_sentence() {
        let text = "Version 3.5 shipped, e.g. on dev vs. prod. J. Smith approved it.";
        assert_eq!(
            chunk(text, 1),
            vec![
                "Version 3.5 shipped, e.g. on dev vs. prod.",
                "J. Smith approved it."
            ]
        );
    }

    #[test]
    fn closing_quotes_stay_with_their_sentence() {
        assert_eq!(
            chunk("It printed \"done.\" Then it exited.", 1),
            vec!["It printed \"done.\"", "Then it exited."]
        );
    }

    #[test]
    fn newlines_end_a_piece_and_empty_pieces_are_dropped() {
        assert_eq!(
            chunk("Three items:\n\n- first\n- second\n", 4),
            vec!["Three items:", "- first", "- second"]
        );
    }

    #[test]
    fn a_runaway_sentence_is_cut_at_a_space() {
        let text = "word ".repeat(80);
        let pieces = chunk(&text, 5);
        assert!(pieces.len() >= 2);
        assert!(pieces.iter().all(|p| p.len() <= MAX_PIECE_CHARS));
        assert!(pieces.iter().all(|p| !p.starts_with(' ')));
    }

    #[test]
    fn markdown_is_said_the_way_a_reader_would_see_it() {
        assert_eq!(speakable("**ISSUE-42** is `open`."), "ISSUE-42 is open.");
        assert_eq!(
            speakable("- see [the PR](https://x.y/1) now"),
            "see the PR now"
        );
        assert_eq!(speakable("## Status"), "Status");
        assert_eq!(speakable("2. second step"), "second step");
        assert_eq!(speakable("run my_script now"), "run my_script now");
        assert_eq!(
            speakable("logs at https://ci.example/run/5"),
            "logs at a link"
        );
        assert_eq!(speakable("_quiet_ please"), "quiet please");
    }
}
