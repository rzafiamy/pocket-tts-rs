pub mod audio;
pub mod conditioners;
pub mod config;
pub mod models;
pub mod modules;
pub mod pause;
pub mod quantize;
pub mod text_chunking;
pub mod tts_model;
pub mod voice_state;
pub mod voices;
pub mod weights;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use pause::{ParsedText, PauseMarker, parse_text_with_pauses};
pub use quantize::{QuantizeConfig, QuantizedTensor};
pub use tts_model::TTSModel;
pub use voice_state::ModelState;
