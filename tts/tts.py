import os
import simpleaudio as sa
from TTS.api import TTS
import torch
from pydub import AudioSegment
from pydub.effects import normalize

# Pick device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {DEVICE}")

# Define the TTS class to load the model once
class TTS_Model:
    def __init__(self, model_name="tts_models/en/vctk/vits", speaker="p234", audio_file="output.wav"):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.AUDIO_FILE = audio_file
        self.SPEAKER = speaker
        
        try:
            self.model = TTS(model_name=model_name, progress_bar=False).to(self.DEVICE)
            print(f"✅ TTS loaded. Using device: {self.DEVICE}")
        except Exception as e:
            print(f"❌ Failed to load TTS: {e}")
            exit()

    def speak(self, text: str, speed: float = 1.3):
        print(f"🗣️ Speaking: {text}")
        
        # Generate speech into file
        self.model.tts_to_file(
            text=text,
            file_path=self.AUDIO_FILE,
            speaker=self.SPEAKER,
            speed=speed
        )

        try:
            # Load the generated audio file
            audio = AudioSegment.from_wav(self.AUDIO_FILE)

            # --- Apply voice modulation effects ---
            # 1. Normalize the volume to a consistent level
            audio = normalize(audio)
            
            # 2. Add a simple reverb/echo effect
            delay_ms = 50 
            audio = audio.overlay(audio, position=delay_ms)

            # 3. Increase the pitch slightly for a more unique tone
            octaves = 0.05
            new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
            
            # 4. Resample the audio to a standard, supported rate
            audio = audio.set_frame_rate(44100)
            
            # Export the modified audio to a new file
            processed_file = "processed_output.wav"
            audio.export(processed_file, format="wav")

            # Play the processed audio
            wave_obj = sa.WaveObject.from_wave_file(processed_file)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        
        except Exception as e:
            print(f"❌ Pydub/Audio Error: {e}")
        
        finally:
            # Clean up files
            if os.path.exists(self.AUDIO_FILE):
                os.remove(self.AUDIO_FILE)
            if os.path.exists(processed_file):
                os.remove(processed_file)