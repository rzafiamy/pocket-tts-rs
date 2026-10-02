//! Text normalization before synthesis, from the `tn` crate
//! ([rzafiamy/tn-rs](https://github.com/rzafiamy/tn-rs)): numbers, times,
//! dates, amounts, units, abbreviations and Markdown become words. Pocket TTS
//! was trained on spelled-out text and reads digits as noise.

pub use tn::{Lang, Lexicon, Mode, normalize, normalize_safe, normalize_text};

/// Language of a model variant (`french`, `french_24l`, `english_2026-04`,
/// `b6369a24`, ...).
pub fn lang_of_variant(variant: &str) -> Lang {
    if variant.is_empty() || variant == "b6369a24" {
        Lang::En
    } else {
        Lang::from_code(variant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// covers: REQ-TXT-003
    #[test]
    fn variant_languages() {
        assert_eq!(lang_of_variant("french_24l"), Lang::Fr);
        assert_eq!(lang_of_variant("english_2026-04"), Lang::En);
        assert_eq!(lang_of_variant("b6369a24"), Lang::En);
        assert_eq!(lang_of_variant("german"), Lang::Other);
    }

    /// covers: REQ-TXT-003
    #[test]
    fn model_languages_are_normalized() {
        assert_eq!(
            normalize("Rendez-vous à 9h30, 12,99 €.", lang_of_variant("french")),
            "Rendez-vous à neuf heures trente, douze euros quatre-vingt-dix-neuf."
        );
        assert_eq!(
            normalize("**Note:** $5.50 at 9:30 am.", lang_of_variant("english")),
            "Note: five dollars and fifty cents at nine thirty a m."
        );
        let plain = "Bonjour, je vous appelle pour le rendez-vous de demain matin.";
        assert_eq!(normalize(plain, Lang::Fr), plain);
    }
}
