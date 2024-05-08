# pip install gTTS
from gtts import gTTS

tts = gTTS('Hello, I am a robot! my name is pedro.')
tts.save('hello.mp3')

##################################################

# pip install openai-whisper
import whisper
model = whisper.load_model("base")
result = model.transcribe("hello.mp3", fp16=False)
print(result["text"])
