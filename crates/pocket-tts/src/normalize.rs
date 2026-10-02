//! Text normalization before synthesis: numbers, times, amounts, units,
//! abbreviations and Markdown become the words a speaker would say.
//!
//! Pocket TTS was trained on spelled-out text and has no front end of its own
//! (neither has the Python reference): "9h30" or "1 250 000 €" come out as
//! noise, and chat-style Markdown (`**bold**`, `## Title`, `- item`) as
//! mumbling. French and English get full number handling; every language gets
//! the Markdown, emoji and line handling.

use regex::{Captures, Regex};
use std::sync::LazyLock;

/// Language of the text rules to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Fr,
    En,
    /// Markdown, emoji and line handling only.
    Other,
}

impl Lang {
    /// Language of a model variant (`french`, `french_24l`, `english_2026-04`,
    /// `b6369a24`, ...).
    pub fn of_variant(variant: &str) -> Self {
        let v = variant.to_ascii_lowercase();
        if v.starts_with("french") {
            Lang::Fr
        } else if v.starts_with("english") || v == "b6369a24" || v.is_empty() {
            Lang::En
        } else {
            Lang::Other
        }
    }
}

/// Normalizes `text` for synthesis in `lang`.
pub fn normalize(text: &str, lang: Lang) -> String {
    let text = markdown_to_sentences(text);
    let text = match lang {
        Lang::Fr => french(&text),
        Lang::En => english(&text),
        Lang::Other => text,
    };
    SPACES.replace_all(text.trim(), " ").into_owned()
}

// ---------------------------------------------------------------------------
// Markdown, emoji, lines
// ---------------------------------------------------------------------------

static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]+").unwrap());
static IMAGE_OR_LINK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!?\[([^\]]*)\]\([^)]*\)").unwrap());
static LINE_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    // headings, quotes, bullets, numbered list items, checkboxes
    Regex::new(r"^(?:#{1,6}\s+|>\s*|[-*+•·]\s+(?:\[[ xX]\]\s+)?|[0-9]{1,3}[.)]\s+)+").unwrap()
});
static EMPHASIS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*+|__+|~~|`+").unwrap());
static SNAKE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\w)_(\w)").unwrap());
static DASHES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+[–—]\s+|[–—]").unwrap());

fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0x2B00..=0x2BFF | 0xFE0F | 0x200D | 0x20E3)
}

/// One sentence per non-empty line: Markdown syntax removed, and a period
/// added where a line (heading, list item) ends without punctuation, so the
/// model pauses there instead of running lines together.
fn markdown_to_sentences(text: &str) -> String {
    let text = IMAGE_OR_LINK.replace_all(text, "$1");
    let mut out = String::with_capacity(text.len());
    let mut in_fence = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || line.is_empty() || is_rule(line) {
            continue;
        }
        let line = if line.starts_with('|') {
            // table row: cells become a list
            line.trim_matches('|')
                .split('|')
                .map(str::trim)
                .filter(|c| !c.is_empty() && !c.chars().all(|ch| matches!(ch, '-' | ':')))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            line.to_string()
        };
        let line = LINE_PREFIX.replace(&line, "");
        let line = EMPHASIS.replace_all(&line, "");
        let line = SNAKE.replace_all(&line, "$1 $2");
        let line: String = line.chars().filter(|c| !is_emoji(*c)).collect();
        let line = DASHES.replace_all(&line, ", ");
        let line = line.trim().trim_end_matches(',').trim();
        if line.is_empty() {
            continue;
        }
        out.push_str(line);
        if !line.ends_with(['.', '!', '?', '…', ':', ';', '»', '"', ')']) {
            out.push('.');
        }
        out.push(' ');
    }
    out
}

fn is_rule(line: &str) -> bool {
    line.len() >= 3
        && line
            .chars()
            .all(|c| matches!(c, '-' | '*' | '_' | '=' | ' '))
}

// ---------------------------------------------------------------------------
// Shared number parsing
// ---------------------------------------------------------------------------

/// A number as written: integer digits and optional decimal digits.
struct Num {
    int: String,
    dec: Option<String>,
}

impl Num {
    /// `thousands`: group separator characters; `decimal`: decimal mark.
    fn parse(s: &str, thousands: &[char], decimal: char) -> Self {
        let (int, dec) = match s.split_once(decimal) {
            Some((i, d)) => (i, Some(d.to_string())),
            None => (s, None),
        };
        Num {
            int: int.chars().filter(|c| !thousands.contains(c)).collect(),
            dec,
        }
    }

    fn value(&self) -> Option<u64> {
        if self.int.len() > 15 {
            return None;
        }
        self.int.parse().ok()
    }

    /// More than one, for plurals.
    fn plural(&self) -> bool {
        self.value().is_none_or(|v| v >= 2)
    }
}

/// Reads a digit string digit by digit.
fn digits(s: &str, names: &[&str; 10]) -> String {
    s.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| names[d as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// French
// ---------------------------------------------------------------------------

const FR_UNITS: [&str; 17] = [
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze",
    "douze", "treize", "quatorze", "quinze", "seize",
];
const FR_DIGITS: [&str; 10] = [
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
];

fn fr_below_100(n: u64) -> String {
    const TENS: [&str; 7] = [
        "",
        "",
        "vingt",
        "trente",
        "quarante",
        "cinquante",
        "soixante",
    ];
    match n {
        0..=16 => FR_UNITS[n as usize].to_string(),
        17..=19 => format!("dix-{}", FR_UNITS[(n - 10) as usize]),
        20..=69 => {
            let (t, u) = (n / 10, n % 10);
            match u {
                0 => TENS[t as usize].to_string(),
                1 => format!("{} et un", TENS[t as usize]),
                _ => format!("{}-{}", TENS[t as usize], FR_UNITS[u as usize]),
            }
        }
        70..=79 => match n {
            71 => "soixante et onze".to_string(),
            _ => format!("soixante-{}", fr_below_100(n - 60)),
        },
        80 => "quatre-vingts".to_string(),
        _ => format!("quatre-vingt-{}", fr_below_100(n - 80)),
    }
}

/// `before_noun`: the number multiplies `mille`, which drops the plural `s` of
/// `cents` and `quatre-vingts` ("deux cent mille", "quatre-vingt mille").
fn fr_below_1000(n: u64, before_mille: bool) -> String {
    let (h, r) = (n / 100, n % 100);
    let mut s = match h {
        0 => return fr_below_100_inv(r, before_mille),
        1 => "cent".to_string(),
        _ => format!("{} cent", FR_UNITS[h as usize]),
    };
    if r == 0 {
        if h > 1 && !before_mille {
            s.push('s');
        }
    } else {
        s.push(' ');
        s.push_str(&fr_below_100_inv(r, before_mille));
    }
    s
}

fn fr_below_100_inv(n: u64, before_mille: bool) -> String {
    if n == 80 && before_mille {
        "quatre-vingt".to_string()
    } else {
        fr_below_100(n)
    }
}

pub(crate) fn fr_cardinal(n: u64) -> String {
    if n == 0 {
        return "zéro".to_string();
    }
    let mut parts = Vec::new();
    let mut rest = n;
    for (scale, one, many) in [
        (1_000_000_000_000u64, "un billion", "billions"),
        (1_000_000_000, "un milliard", "milliards"),
        (1_000_000, "un million", "millions"),
    ] {
        let q = rest / scale;
        rest %= scale;
        if q == 1 {
            parts.push(one.to_string());
        } else if q > 1 {
            parts.push(format!("{} {many}", fr_cardinal(q)));
        }
    }
    let thousands = rest / 1000;
    rest %= 1000;
    if thousands == 1 {
        parts.push("mille".to_string());
    } else if thousands > 1 {
        parts.push(format!("{} mille", fr_below_1000(thousands, true)));
    }
    if rest > 0 {
        parts.push(fr_below_1000(rest, false));
    }
    parts.join(" ")
}

/// Feminine form ("une heure", "vingt et une heures").
fn fr_cardinal_fem(n: u64) -> String {
    let s = fr_cardinal(n);
    match s.strip_suffix("un") {
        Some(head) if head.is_empty() || head.ends_with(' ') || head.ends_with('-') => {
            format!("{head}une")
        }
        _ => s,
    }
}

fn fr_ordinal(n: u64, feminine: bool) -> String {
    if n == 1 {
        return if feminine { "première" } else { "premier" }.to_string();
    }
    let card = fr_cardinal(n);
    let card = card.strip_suffix('s').unwrap_or(&card).to_string(); // cents, vingts
    if let Some(h) = card.strip_suffix("cinq") {
        format!("{h}cinquième")
    } else if let Some(h) = card.strip_suffix("neuf") {
        format!("{h}neuvième")
    } else if let Some(h) = card.strip_suffix('e') {
        format!("{h}ième")
    } else {
        format!("{card}ième")
    }
}

/// A number as read in French: integers, decimals ("virgule"), leading zeros
/// and very long strings digit by digit.
fn fr_number(num: &Num) -> String {
    let int = match num.value() {
        Some(v) if !(num.int.len() > 1 && num.int.starts_with('0')) => fr_cardinal(v),
        _ => digits(&num.int, &FR_DIGITS),
    };
    match &num.dec {
        Some(d) if !d.is_empty() => format!("{int} virgule {}", fr_decimal(d)),
        _ => int,
    }
}

fn fr_decimal(d: &str) -> String {
    let zeros = d.chars().take_while(|c| *c == '0').count();
    let rest = &d[zeros..];
    let mut words: Vec<String> = std::iter::repeat_n("zéro".to_string(), zeros).collect();
    if !rest.is_empty() {
        words.push(if rest.len() <= 3 {
            fr_cardinal(rest.parse().unwrap_or(0))
        } else {
            digits(rest, &FR_DIGITS)
        });
    }
    words.join(" ")
}

/// "d'euros" after an exact million / milliard, else " euros".
fn fr_of(amount: &str, noun: &str) -> String {
    let elided = amount.ends_with("million")
        || amount.ends_with("millions")
        || amount.ends_with("milliard")
        || amount.ends_with("milliards");
    if elided {
        let vowel = noun.starts_with(['a', 'e', 'é', 'i', 'o', 'u']);
        format!("{amount} {}{noun}", if vowel { "d'" } else { "de " })
    } else {
        format!("{amount} {noun}")
    }
}

// Regex pieces. FR numbers: "1 250 000", "1250000", "3,5"; separators include
// the no-break spaces typography puts there.
const FR_NUM: &str = r"[0-9]{1,3}(?:[ \u{a0}\u{202f}][0-9]{3})+(?:,[0-9]+)?|[0-9]+(?:,[0-9]+)?";
const FR_SEP: [char; 3] = [' ', '\u{a0}', '\u{202f}'];

static FR_ABBR: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"\bMmes\b\.?", "Mesdames"),
        (r"\bMme\b\.?", "Madame"),
        (r"\bMlles\b\.?", "Mesdemoiselles"),
        (r"\bMlle\b\.?", "Mademoiselle"),
        (r"\bMM\.", "Messieurs"),
        (r"\bM\.\s", "Monsieur "),
        (r"\bDr\b\.?", "Docteur"),
        (r"\bPr\b\.?", "Professeur"),
        (r"\bc\.-à-d\.", "c'est-à-dire"),
        (r"\bp\. ?ex\.", "par exemple"),
        (r"\benv\.", "environ"),
        (r"\betc\.", "et cetera."),
        (r"\b[nN]°\s?", "numéro "),
        (r"\s&\s", " et "),
        (r"\s\+\s", " plus "),
        (r"\s=\s", " égale "),
        (r"@", " arobase "),
    ]
    .into_iter()
    .map(|(p, r)| (Regex::new(p).unwrap(), r))
    .collect()
});

static FR_TIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b([0-9]{1,2})\s?(?:h|H|:)\s?([0-9]{2})\b|\b([0-9]{1,2})\s?h\b").unwrap()
});
static FR_ORDINAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([0-9]+)(er|re|ère|ere|ᵉʳ|ᵉ|e|ème|eme|nd|nde)\b").unwrap());
static FR_MONEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?:([€$£])\s?({FR_NUM}))|(?:({FR_NUM})\s?(€|\$|£|EUR\b|USD\b))"
    ))
    .unwrap()
});
static FR_PERCENT_DEGREE_UNIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(^|[\s(])?(-)?({FR_NUM})\s?(%|°\s?C\b|°\s?F\b|°|km/h\b|km\b|kg\b|cm\b|mm\b|m²|m2\b|m³|kWh\b|kW\b|Go\b|Mo\b|min\b)"
    ))
    .unwrap()
});
static FR_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"(^|[\s(])?(-)?({FR_NUM})")).unwrap());

fn french(text: &str) -> String {
    let mut s = text.to_string();
    for (re, rep) in FR_ABBR.iter() {
        s = re.replace_all(&s, *rep).into_owned();
    }
    s = FR_TIME
        .replace_all(&s, |c: &Captures| {
            if let (Some(h), Some(m)) = (c.get(1), c.get(2)) {
                let h: u64 = h.as_str().parse().unwrap_or(0);
                let m: u64 = m.as_str().parse().unwrap_or(0);
                let hours = format!(
                    "{} {}",
                    fr_cardinal_fem(h),
                    if h >= 2 { "heures" } else { "heure" }
                );
                if m == 0 {
                    hours
                } else {
                    format!("{hours} {}", fr_cardinal_fem(m))
                }
            } else {
                let h: u64 = c[3].parse().unwrap_or(0);
                format!(
                    "{} {}",
                    fr_cardinal_fem(h),
                    if h >= 2 { "heures" } else { "heure" }
                )
            }
        })
        .into_owned();
    s = FR_ORDINAL
        .replace_all(&s, |c: &Captures| {
            let n: u64 = c[1].parse().unwrap_or(0);
            let fem = matches!(&c[2], "re" | "ère" | "nde");
            if matches!(&c[2], "nd" | "nde") && n == 2 {
                return if fem { "seconde" } else { "second" }.to_string();
            }
            fr_ordinal(n, fem)
        })
        .into_owned();
    s = FR_MONEY
        .replace_all(&s, |c: &Captures| {
            let (sym, raw) = match (c.get(1), c.get(2)) {
                (Some(sym), Some(n)) => (sym.as_str(), n.as_str()),
                _ => (&c[4], &c[3]),
            };
            let (one, many) = match sym {
                "€" | "EUR" => ("euro", "euros"),
                "$" | "USD" => ("dollar", "dollars"),
                _ => ("livre", "livres"),
            };
            let num = Num::parse(raw, &FR_SEP, ',');
            let int_num = Num {
                int: num.int.clone(),
                dec: None,
            };
            let amount = fr_number(&int_num);
            let noun = if int_num.plural() { many } else { one };
            let main = fr_of(&amount, noun);
            match num.dec.as_deref() {
                Some(d) if (1..=2).contains(&d.len()) => {
                    let cents: u64 = format!("{d:0<2}").parse().unwrap_or(0);
                    if cents == 0 {
                        main
                    } else {
                        format!("{main} {}", fr_cardinal(cents))
                    }
                }
                Some(d) if !d.is_empty() => format!("{} {many}", fr_number(&num)),
                _ => main,
            }
        })
        .into_owned();
    s = FR_PERCENT_DEGREE_UNIT
        .replace_all(&s, |c: &Captures| {
            let lead = c.get(1).map_or("", |m| m.as_str());
            let minus = if c.get(2).is_some() { "moins " } else { "" };
            let num = Num::parse(&c[3], &FR_SEP, ',');
            let words = fr_number(&num);
            let pl = num.plural() || num.dec.is_some();
            let unit = match c[4].replace(' ', "").as_str() {
                "%" => "pour cent".to_string(),
                "°C" | "°" => if pl { "degrés" } else { "degré" }.to_string(),
                "°F" => format!("{} Fahrenheit", if pl { "degrés" } else { "degré" }),
                "km/h" => "kilomètres heure".to_string(),
                "km" => plural_fr("kilomètre", pl),
                "kg" => plural_fr("kilo", pl),
                "cm" => plural_fr("centimètre", pl),
                "mm" => plural_fr("millimètre", pl),
                "m²" | "m2" => format!(
                    "{} carré{}",
                    plural_fr("mètre", pl),
                    if pl { "s" } else { "" }
                ),
                "m³" => format!(
                    "{} cube{}",
                    plural_fr("mètre", pl),
                    if pl { "s" } else { "" }
                ),
                "kWh" => plural_fr("kilowattheure", pl),
                "kW" => plural_fr("kilowatt", pl),
                "Go" => plural_fr("gigaoctet", pl),
                "Mo" => plural_fr("mégaoctet", pl),
                "min" => plural_fr("minute", pl),
                other => other.to_string(),
            };
            format!("{lead}{minus}{words} {unit}")
        })
        .into_owned();
    s = FR_NUMBER
        .replace_all(&s, |c: &Captures| {
            let lead = c.get(1).map_or("", |m| m.as_str());
            // a minus sign only at a word start ("-5", "(-3"), not "Covid-19"
            let minus =
                if c.get(2).is_some() && (c.get(1).is_some() || c.get(0).unwrap().start() == 0) {
                    "moins "
                } else if c.get(2).is_some() {
                    "-"
                } else {
                    ""
                };
            let num = Num::parse(&c[3], &FR_SEP, ',');
            format!("{lead}{minus}{}", fr_number(&num))
        })
        .into_owned();
    s
}

fn plural_fr(word: &str, plural: bool) -> String {
    if plural {
        format!("{word}s")
    } else {
        word.to_string()
    }
}

// ---------------------------------------------------------------------------
// English
// ---------------------------------------------------------------------------

const EN_ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const EN_TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const EN_DIGITS: [&str; 10] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];

fn en_below_100(n: u64) -> String {
    if n < 20 {
        EN_ONES[n as usize].to_string()
    } else if n.is_multiple_of(10) {
        EN_TENS[(n / 10) as usize].to_string()
    } else {
        format!(
            "{}-{}",
            EN_TENS[(n / 10) as usize],
            EN_ONES[(n % 10) as usize]
        )
    }
}

fn en_below_1000(n: u64) -> String {
    let (h, r) = (n / 100, n % 100);
    match (h, r) {
        (0, _) => en_below_100(r),
        (_, 0) => format!("{} hundred", EN_ONES[h as usize]),
        _ => format!("{} hundred {}", EN_ONES[h as usize], en_below_100(r)),
    }
}

pub(crate) fn en_cardinal(n: u64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    let mut parts = Vec::new();
    let mut rest = n;
    for (scale, name) in [
        (1_000_000_000_000u64, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ] {
        let q = rest / scale;
        rest %= scale;
        if q > 0 {
            parts.push(format!("{} {name}", en_cardinal(q)));
        }
    }
    if rest > 0 {
        parts.push(en_below_1000(rest));
    }
    parts.join(" ")
}

fn en_ordinal(n: u64) -> String {
    let card = en_cardinal(n);
    let (head, last) = match card.rfind([' ', '-']) {
        Some(i) => (&card[..=i], &card[i + 1..]),
        None => ("", card.as_str()),
    };
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    format!("{head}{last}")
}

/// A four-digit year read as a year ("nineteen eighty-four", "twenty
/// twenty-seven", "two thousand five").
fn en_year(n: u64) -> String {
    let (hi, lo) = (n / 100, n % 100);
    if (2000..2010).contains(&n) {
        return if lo == 0 {
            "two thousand".into()
        } else {
            format!("two thousand {}", en_below_100(lo))
        };
    }
    match lo {
        0 => format!("{} hundred", en_below_100(hi)),
        1..=9 => format!("{} oh {}", en_below_100(hi), EN_ONES[lo as usize]),
        _ => format!("{} {}", en_below_100(hi), en_below_100(lo)),
    }
}

fn en_number(num: &Num, years: bool) -> String {
    let int = match num.value() {
        Some(_) if num.int.len() > 1 && num.int.starts_with('0') => digits(&num.int, &EN_DIGITS),
        Some(v)
            if years && num.dec.is_none() && num.int.len() == 4 && (1100..2100).contains(&v) =>
        {
            en_year(v)
        }
        Some(v) => en_cardinal(v),
        None => digits(&num.int, &EN_DIGITS),
    };
    match &num.dec {
        Some(d) if !d.is_empty() => format!("{int} point {}", digits(d, &EN_DIGITS)),
        _ => int,
    }
}

const EN_NUM: &str = r"[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?";
const EN_SEP: [char; 1] = [','];

static EN_ABBR: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"\bMr\.", "Mister"),
        (r"\bMrs\.", "Missus"),
        (r"\bMs\.", "Miz"),
        (r"\bDr\.", "Doctor"),
        (r"\bProf\.", "Professor"),
        (r"\be\.g\.", "for example"),
        (r"\bi\.e\.", "that is"),
        (r"\betc\.", "et cetera."),
        (r"\bvs\.?", "versus"),
        (r"\bNo\.\s?([0-9])", "number $1"),
        (r"#([0-9])", "number $1"),
        (r"\s&\s", " and "),
        (r"\s\+\s", " plus "),
        (r"\s=\s", " equals "),
        (r"@", " at "),
    ]
    .into_iter()
    .map(|(p, r)| (Regex::new(p).unwrap(), r))
    .collect()
});

static EN_TIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b([0-9]{1,2})(?::([0-9]{2}))?\s?([aApP])\.?[mM]\b|\b([0-9]{1,2}):([0-9]{2})\b")
        .unwrap()
});
static EN_ORDINAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([0-9]+)(st|nd|rd|th)\b").unwrap());
static EN_MONEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?:([€$£])\s?({EN_NUM})(?:\s?(k|K|million|billion)\b)?)|(?:({EN_NUM})\s?(€|EUR\b|USD\b))"
    ))
    .unwrap()
});
static EN_PERCENT_DEGREE_UNIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(^|[\s(])?(-)?({EN_NUM})\s?(%|°\s?C\b|°\s?F\b|°|km/h\b|mph\b|km\b|kg\b|cm\b|mm\b|lbs?\b|ft\b|kWh\b|GB\b|MB\b|min\b)"
    ))
    .unwrap()
});
static EN_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"(^|[\s(])?(-)?({EN_NUM})")).unwrap());

fn english(text: &str) -> String {
    let mut s = text.to_string();
    for (re, rep) in EN_ABBR.iter() {
        s = re.replace_all(&s, *rep).into_owned();
    }
    s = EN_TIME
        .replace_all(&s, |c: &Captures| {
            let (h, m, ampm) = match c.get(1) {
                Some(h) => (
                    h.as_str(),
                    c.get(2).map(|m| m.as_str()),
                    c.get(3).map(|a| a.as_str()),
                ),
                None => (&c[4], c.get(5).map(|m| m.as_str()), None),
            };
            let h: u64 = h.parse().unwrap_or(0);
            let mut out = en_cardinal(h);
            match m.map(|m| m.parse::<u64>().unwrap_or(0)) {
                Some(0) if ampm.is_none() => out.push_str(" o'clock"),
                Some(0) | None => {}
                Some(m) if m < 10 => out.push_str(&format!(" oh {}", EN_ONES[m as usize])),
                Some(m) => out.push_str(&format!(" {}", en_below_100(m))),
            }
            if let Some(a) = ampm {
                out.push_str(if a.eq_ignore_ascii_case("a") {
                    " a m"
                } else {
                    " p m"
                });
            }
            out
        })
        .into_owned();
    s = EN_ORDINAL
        .replace_all(&s, |c: &Captures| en_ordinal(c[1].parse().unwrap_or(0)))
        .into_owned();
    s = EN_MONEY
        .replace_all(&s, |c: &Captures| {
            let (sym, raw, scale) = match (c.get(1), c.get(2)) {
                (Some(sym), Some(n)) => (sym.as_str(), n.as_str(), c.get(3).map(|m| m.as_str())),
                _ => (&c[5], &c[4], None),
            };
            let (one, many, cent_one, cent_many) = match sym {
                "€" | "EUR" => ("euro", "euros", "cent", "cents"),
                "$" | "USD" => ("dollar", "dollars", "cent", "cents"),
                _ => ("pound", "pounds", "penny", "pence"),
            };
            let num = Num::parse(raw, &EN_SEP, '.');
            if let Some(scale) = scale {
                let scale = if scale.eq_ignore_ascii_case("k") {
                    "thousand"
                } else {
                    scale
                };
                return format!("{} {scale} {many}", en_number(&num, false));
            }
            let int_num = Num {
                int: num.int.clone(),
                dec: None,
            };
            let main = format!(
                "{} {}",
                en_number(&int_num, false),
                if int_num.plural() { many } else { one }
            );
            match num.dec.as_deref() {
                Some(d) if (1..=2).contains(&d.len()) => {
                    let cents: u64 = format!("{d:0<2}").parse().unwrap_or(0);
                    match cents {
                        0 => main,
                        1 => format!("{main} and one {cent_one}"),
                        _ => format!("{main} and {} {cent_many}", en_cardinal(cents)),
                    }
                }
                Some(d) if !d.is_empty() => format!("{} {many}", en_number(&num, false)),
                _ => main,
            }
        })
        .into_owned();
    s = EN_PERCENT_DEGREE_UNIT
        .replace_all(&s, |c: &Captures| {
            let lead = c.get(1).map_or("", |m| m.as_str());
            let minus = if c.get(2).is_some() { "minus " } else { "" };
            let num = Num::parse(&c[3], &EN_SEP, '.');
            let words = en_number(&num, false);
            let pl = num.plural() || num.dec.is_some();
            let unit = match c[4].replace(' ', "").as_str() {
                "%" => "percent".to_string(),
                "°C" => format!("{} Celsius", if pl { "degrees" } else { "degree" }),
                "°F" => format!("{} Fahrenheit", if pl { "degrees" } else { "degree" }),
                "°" => if pl { "degrees" } else { "degree" }.to_string(),
                "km/h" => "kilometers per hour".to_string(),
                "mph" => "miles per hour".to_string(),
                "km" => plural_en("kilometer", pl),
                "kg" => plural_en("kilogram", pl),
                "cm" => plural_en("centimeter", pl),
                "mm" => plural_en("millimeter", pl),
                "lb" | "lbs" => plural_en("pound", pl),
                "ft" => if pl { "feet" } else { "foot" }.to_string(),
                "kWh" => plural_en("kilowatt hour", pl),
                "GB" => plural_en("gigabyte", pl),
                "MB" => plural_en("megabyte", pl),
                "min" => plural_en("minute", pl),
                other => other.to_string(),
            };
            format!("{lead}{minus}{words} {unit}")
        })
        .into_owned();
    s = EN_NUMBER
        .replace_all(&s, |c: &Captures| {
            let lead = c.get(1).map_or("", |m| m.as_str());
            let minus =
                if c.get(2).is_some() && (c.get(1).is_some() || c.get(0).unwrap().start() == 0) {
                    "minus "
                } else if c.get(2).is_some() {
                    "-"
                } else {
                    ""
                };
            let num = Num::parse(&c[3], &EN_SEP, '.');
            format!("{lead}{minus}{}", en_number(&num, true))
        })
        .into_owned();
    s
}

fn plural_en(word: &str, plural: bool) -> String {
    if plural {
        format!("{word}s")
    } else {
        word.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(s: &str) -> String {
        normalize(s, Lang::Fr)
    }
    fn en(s: &str) -> String {
        normalize(s, Lang::En)
    }

    /// covers: REQ-TXT-003
    #[test]
    fn french_cardinals() {
        let cases = [
            (0, "zéro"),
            (1, "un"),
            (16, "seize"),
            (17, "dix-sept"),
            (21, "vingt et un"),
            (22, "vingt-deux"),
            (70, "soixante-dix"),
            (71, "soixante et onze"),
            (77, "soixante-dix-sept"),
            (80, "quatre-vingts"),
            (81, "quatre-vingt-un"),
            (91, "quatre-vingt-onze"),
            (99, "quatre-vingt-dix-neuf"),
            (100, "cent"),
            (101, "cent un"),
            (200, "deux cents"),
            (201, "deux cent un"),
            (1000, "mille"),
            (1990, "mille neuf cent quatre-vingt-dix"),
            (2027, "deux mille vingt-sept"),
            (80_000, "quatre-vingt mille"),
            (200_000, "deux cent mille"),
            (1_250_000, "un million deux cent cinquante mille"),
            (2_000_000, "deux millions"),
            (3_000_000_000, "trois milliards"),
        ];
        for (n, want) in cases {
            assert_eq!(fr_cardinal(n), want, "{n}");
        }
    }

    /// covers: REQ-TXT-003
    #[test]
    fn french_text() {
        assert_eq!(
            fr("Rendez-vous à 9h30."),
            "Rendez-vous à neuf heures trente."
        );
        assert_eq!(fr("À 14h, puis 1h."), "À quatorze heures, puis une heure.");
        assert_eq!(fr("Départ à 21:05."), "Départ à vingt et une heures cinq.");
        assert_eq!(
            fr("Budget de 1 250 000 €."),
            "Budget de un million deux cent cinquante mille euros."
        );
        assert_eq!(fr("Total : 2 000 000 €."), "Total : deux millions d'euros.");
        assert_eq!(fr("Prix : 5,50 €."), "Prix : cinq euros cinquante.");
        assert_eq!(
            fr("Il fera 18 °C, puis -3°C."),
            "Il fera dix-huit degrés, puis moins trois degrés."
        );
        assert_eq!(
            fr("Hausse de 3,5 %."),
            "Hausse de trois virgule cinq pour cent."
        );
        assert_eq!(
            fr("Le 1er mai, la 2e fois, le 21e siècle."),
            "Le premier mai, la deuxième fois, le vingt et unième siècle."
        );
        assert_eq!(
            fr("Appelez le 06 12 34 56 78."),
            "Appelez le zéro six douze trente-quatre cinquante-six soixante-dix-huit."
        );
        assert_eq!(
            fr("Mme Dupont et M. Martin."),
            "Madame Dupont et Monsieur Martin."
        );
        assert_eq!(
            fr("Le Covid-19 en 2020."),
            "Le Covid-dix-neuf en deux mille vingt."
        );
        assert_eq!(
            fr("À 120 km/h sur 3 km."),
            "À cent vingt kilomètres heure sur trois kilomètres."
        );
    }

    /// covers: REQ-TXT-003
    #[test]
    fn english_text() {
        assert_eq!(en("Meet at 9:30 am."), "Meet at nine thirty a m.");
        assert_eq!(en("At 5 PM or 10:05."), "At five p m or ten oh five.");
        assert_eq!(
            en("It costs $5.50."),
            "It costs five dollars and fifty cents."
        );
        assert_eq!(en("A $2 million deal."), "A two million dollars deal.");
        assert_eq!(
            en("Up 12% to 1,250,000."),
            "Up twelve percent to one million two hundred fifty thousand."
        );
        assert_eq!(
            en("In 1984 and 2027."),
            "In nineteen eighty-four and twenty twenty-seven."
        );
        assert_eq!(en("The 21st and 3rd."), "The twenty-first and third.");
        assert_eq!(
            en("It is 72°F, 3.14."),
            "It is seventy-two degrees Fahrenheit, three point one four."
        );
        assert_eq!(
            en("Dr. Smith & Mr. Jones."),
            "Doctor Smith and Mister Jones."
        );
    }

    /// covers: REQ-TXT-003
    #[test]
    fn markdown() {
        let md = "## Votre semaine\n\nVoici :\n\n1. **Réunion** lundi\n- *Livraison* jeudi 🚚\n---\n| Jour | Lieu |\n|---|---|\n| Lundi | Paris |\n```\ncode\n```\nVoir [le site](https://x.fr).";
        assert_eq!(
            normalize(md, Lang::Other),
            "Votre semaine. Voici : Réunion lundi. Livraison jeudi. Jour, Lieu. Lundi, Paris. Voir le site."
        );
    }

    /// covers: REQ-TXT-003
    #[test]
    fn plain_text_unchanged() {
        let s = "Bonjour, je vous appelle pour le rendez-vous de demain matin.";
        assert_eq!(fr(s), s);
        let s = "Hello, this is a plain sentence; nothing to change.";
        assert_eq!(en(s), s);
    }

    #[test]
    fn variant_languages() {
        assert_eq!(Lang::of_variant("french_24l"), Lang::Fr);
        assert_eq!(Lang::of_variant("english_2026-04"), Lang::En);
        assert_eq!(Lang::of_variant("b6369a24"), Lang::En);
        assert_eq!(Lang::of_variant("german"), Lang::Other);
    }
}
