import whisper_at as whisper

audio_tagging_time_resolution = 10
model = whisper.load_model("large-v1")
audio_tagging_time_resolution = 10
result = model.transcribe("./audio.mp3", at_time_res=audio_tagging_time_resolution)
print(result['text'])