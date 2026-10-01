import sys
from pocket_tts import TTSModel
from pocket_tts.models.text_chunking import split_into_best_sentences, prepare_text_prompt
m = TTSModel.load_model(language=sys.argv[1])
tok = m.flow_lm.conditioner.tokenizer
opts = dict(remove_semicolons=m.remove_semicolons, append_terminal_punctuation=m.append_terminal_punctuation, capitalize_first_letter=m.capitalize_first_letter, replace_characters=m.replace_characters)
for c in split_into_best_sentences(tok, sys.argv[2], 50, m.pad_with_spaces_for_short_inputs, **opts):
    p, g = prepare_text_prompt(c, m.pad_with_spaces_for_short_inputs, **opts)
    print(repr(p), f"guess={g}", "ids=" + str(tok.encode(p)).replace(" ", ""))
