//! Predefined voices: per-language model states precomputed by Kyutai
//! (Python's `utils.get_predefined_voice` and `DEFAULT_VOICE_FOR_LANGUAGE`).

/// Voice names available for every released language model.
pub const PREDEFINED_VOICES: &[&str] = &[
    "cosette",
    "marius",
    "javert",
    "alba",
    "jean",
    "anna",
    "vera",
    "fantine",
    "charles",
    "paul",
    "eponine",
    "azelma",
    "george",
    "mary",
    "jane",
    "michael",
    "eve",
    "bill_boerst",
    "peter_yearsley",
    "stuart_bell",
    "caro_davy",
    "giovanni",
    "lola",
    "juergen",
    "rafael",
    "daan",
    "estelle",
];

/// Voices of the original `b6369a24` English model (latent prompts).
const LEGACY_VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
];
const LEGACY_VARIANT: &str = "b6369a24";
const VOICES_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";
const VOICES_REVISION: &str = "1e08e6a23401048648a9fdcfde2f89348215c2a7";

/// Whether `name` is a predefined voice of model `variant`.
pub fn is_predefined(variant: &str, name: &str) -> bool {
    if variant == LEGACY_VARIANT {
        LEGACY_VOICES.contains(&name)
    } else {
        PREDEFINED_VOICES.contains(&name)
    }
}

/// `hf://` location of predefined voice `name` for model `variant`.
pub fn predefined_voice_url(variant: &str, name: &str) -> String {
    if variant == LEGACY_VARIANT {
        format!("hf://{VOICES_REPO}/embeddings/{name}.safetensors")
    } else {
        // Variants are named after their language folder (`french`, `english_2026-09_24l`).
        format!(
            "hf://{VOICES_REPO}/languages/{variant}/embeddings/{name}.safetensors@{VOICES_REVISION}"
        )
    }
}

/// The voice used when none is given: a native speaker for each language.
pub fn default_voice(variant: &str) -> &'static str {
    let language = variant.split('_').next().unwrap_or(variant);
    match language {
        "italian" => "giovanni",
        "spanish" => "lola",
        "german" => "juergen",
        "portuguese" => "rafael",
        "french" => "estelle",
        "dutch" => "daan",
        _ => "alba",
    }
}

/// Greeting used when no text is given (Python's `DEFAULT_TEXT_FOR_LANGUAGE`).
pub fn default_text(variant: &str) -> &'static str {
    let language = variant.split('_').next().unwrap_or(variant);
    match language {
        "french" => {
            "Bonjour le monde. Je suis le TTS de poche de Kyutai. \
             Je suis assez rapide pour fonctionner sur de petits CPU. \
             J'espère que vous m'aimerez."
        }
        "german" => {
            "Hallo Welt. Ich bin Pocket TTS von Kyutai. \
             Ich bin schnell genug, um auch auf kleinen CPUs zu laufen. \
             Ich hoffe, ich gefalle dir."
        }
        "portuguese" => {
            "Olá mundo. Eu sou o Pocket TTS da Kyutai. \
             Sou rápido o suficiente para rodar em CPUs pequenas. \
             Espero que você goste de mim."
        }
        "italian" => {
            "Ciao mondo. Sono il Pocket TTS di Kyutai. \
             Sono abbastanza veloce da funzionare su piccole CPU. \
             Spero che ti piacerò."
        }
        "dutch" => {
            "Hallo wereld. Ik ben Pocket TTS van Kyutai. \
             Ik ben snel genoeg om op kleine CPU's te draaien. \
             Ik hoop dat je me leuk vindt."
        }
        "spanish" => {
            "Hola mundo. Soy el Pocket TTS de Kyutai. \
             Soy lo suficientemente rápido para funcionar en pequeñas CPU. \
             Espero que te guste."
        }
        _ => {
            "Hello world. I am Kyutai's Pocket TTS. \
             I'm fast enough to run on small CPUs. \
             I hope you'll like me."
        }
    }
}
