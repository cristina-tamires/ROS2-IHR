# pip install gTTS
from gtts import gTTS

tts = gTTS('Olá, eu sou um robô! Meu nome é Maria. Como posso te ajudar?', lang ="pt",)
tts.save('hello.mp3')

# pip install python-vlc
import vlc
p = vlc.MediaPlayer("hello.mp3")
p.play()

##################################################

# pip install openai-whisper
import whisper
model = whisper.load_model("base")
result = model.transcribe("hello.mp3", fp16=False)
print(result["text"])
