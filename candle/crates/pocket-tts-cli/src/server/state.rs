use pocket_tts::TTSModel;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<TTSModel>,
    // Lock to ensure sequential processing of generation requests
    // (Matching Python's "not thread safe" / single worker behavior)
    pub lock: Arc<Mutex<()>>,
}

impl AppState {
    pub fn new(model: TTSModel) -> Self {
        Self {
            model: Arc::new(model),
            lock: Arc::new(Mutex::new(())),
        }
    }
}
