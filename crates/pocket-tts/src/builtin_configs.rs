//! Upstream configs compiled into the binary, keyed by variant name.

pub const CONFIGS: &[(&str, &str)] = &[
    ("b6369a24", include_str!("../config/b6369a24.yaml")),
    ("dutch", include_str!("../config/dutch.yaml")),
    ("dutch_24l", include_str!("../config/dutch_24l.yaml")),
    ("english", include_str!("../config/english.yaml")),
    (
        "english_2026-01",
        include_str!("../config/english_2026-01.yaml"),
    ),
    (
        "english_2026-04",
        include_str!("../config/english_2026-04.yaml"),
    ),
    (
        "english_2026-04_24l",
        include_str!("../config/english_2026-04_24l.yaml"),
    ),
    (
        "english_2026-09",
        include_str!("../config/english_2026-09.yaml"),
    ),
    (
        "english_2026-09_24l",
        include_str!("../config/english_2026-09_24l.yaml"),
    ),
    (
        "english_drifting_26-09",
        include_str!("../config/english_drifting_26-09.yaml"),
    ),
    ("french", include_str!("../config/french.yaml")),
    ("french_24l", include_str!("../config/french_24l.yaml")),
    ("german", include_str!("../config/german.yaml")),
    ("german_24l", include_str!("../config/german_24l.yaml")),
    ("italian", include_str!("../config/italian.yaml")),
    ("italian_24l", include_str!("../config/italian_24l.yaml")),
    ("portuguese", include_str!("../config/portuguese.yaml")),
    (
        "portuguese_24l",
        include_str!("../config/portuguese_24l.yaml"),
    ),
    ("spanish", include_str!("../config/spanish.yaml")),
    ("spanish_24l", include_str!("../config/spanish_24l.yaml")),
];

/// YAML of the built-in config `variant`.
pub fn get(variant: &str) -> Option<&'static str> {
    CONFIGS.iter().find(|(n, _)| *n == variant).map(|(_, y)| *y)
}

/// Names of the built-in variants.
pub fn names() -> impl Iterator<Item = &'static str> {
    CONFIGS.iter().map(|(n, _)| *n)
}
