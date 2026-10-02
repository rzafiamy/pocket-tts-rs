//! Prints `text` as normalized before synthesis (`normalize.rs`).
//! Usage: cargo run --release --example normalize -- [--safe] <variant|fr|en> <text>
//! `--safe` leaves ambiguous numbers as digits (`normalize_safe`).
use pocket_tts::normalize::{Lang, normalize, normalize_safe};

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let safe = args.get(1).is_some_and(|a| a == "--safe");
    if safe {
        args.remove(1);
    }
    let lang = match args.get(1).map(String::as_str) {
        Some("fr") => Lang::Fr,
        Some("en") => Lang::En,
        Some(v) => Lang::of_variant(v),
        None => Lang::En,
    };
    let text = args[2..].join(" ");
    println!(
        "{}",
        if safe {
            normalize_safe(&text, lang)
        } else {
            normalize(&text, lang)
        }
    );
}
