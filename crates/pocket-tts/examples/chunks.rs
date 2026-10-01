//! Prints the chunks, prepared prompts and token ids for a text.
//! Usage: cargo run --release --example chunks -- <variant> <text>
use pocket_tts::TTSModel;
use pocket_tts::text_chunking::prepare_text_prompt;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = TTSModel::load(&args[1])?;
    for chunk in model.split_into_best_sentences(&args[2])? {
        let (prepared, guess) = prepare_text_prompt(&chunk, &model.text_options())?;
        let ids = model.conditioner.encode(&prepared)?;
        println!("{prepared:?} guess={guess} ids={ids:?}");
    }
    Ok(())
}
