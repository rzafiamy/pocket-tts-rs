import sys, torch, scipy.io.wavfile as w
from pocket_tts import TTSModel
lang, voice, text, out = sys.argv[1:5]
temp = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
torch.manual_seed(0)
m = TTSModel.load_model(language=lang, temp=temp)
import os
if os.environ.get("FPC"): m.max_decoder_frames_per_call = int(os.environ["FPC"])
st = m.get_state_for_audio_prompt(voice)
a = m.generate_audio(st, text)
w.write(out, m.sample_rate, a.numpy())
print("samples", a.shape)
