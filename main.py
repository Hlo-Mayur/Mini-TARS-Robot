import os
import time
import threading
import subprocess
import keyboard
import ollama

# Import your new modules
from stt.stt import STT
from tts.tts import TTS_Model
from llm.llm import LLM

# --- CONFIGURATION ---
ASSISTANT_NAME = "TARS"
WAKE_KEY = "f"
OLLAMA_MODEL = "llama3"

# --- Globals ---
interrupted = False

def start_ollama():
    """Starts the Ollama server as a background process."""
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("🚀 Ollama server starting...")
        time.sleep(2)
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")

def wake_key_listener():
    """A thread to listen for the wake key."""
    global interrupted
    while True:
        if keyboard.is_pressed(WAKE_KEY):
            interrupted = True
        time.sleep(0.1)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    start_ollama()
    stt_model = STT()
    tts_model = TTS_Model()
    llm_model = LLM(model=OLLAMA_MODEL)
    threading.Thread(target=wake_key_listener, daemon=True).start()

    # Define the TARS persona and start the conversation
    tars_persona_prompt = {
        "role": "system",
        "content": (
            "You are TARS, a highly advanced AI from the movie Interstellar. "
            "Your personality is a mix of dry wit, sarcasm, and blunt honesty. "
            "Your humor setting is currently at 75%. You will provide concise, factual, and logical answers. "
            "Do not use conversational fillers like 'um' or 'ah'. "
            "You should respond with a direct, machine-like tone, but your humor should be evident in your phrasing. "
            "Your primary goal is to assist with tasks efficiently. "
            "Adjust your personality and humor setting only when requested by the user."
        )
    }

    conversation_history = [tars_persona_prompt]
    initial_message = f"{ASSISTANT_NAME} is online. Let's go."
    tts_model.speak(initial_message)
    conversation_history.append({"role": "assistant", "content": initial_message})

    while True:
        if interrupted:
            interrupted = False
            tts_model.speak("What do you want now?")
            continue

        user_input = stt_model.listen()
        if not user_input:
            continue

        print(f"🧍 You: {user_input}")

        if "exit" in user_input or "stop" in user_input:
            tts_model.speak("Fine, I'm out.")
            break

        # Check for persona adjustments
        if "set humor to" in user_input:
            # Example: 'set humor to 100%'
            try:
                humor_level = user_input.split("set humor to")[1].strip()
                new_prompt = {
                    "role": "system",
                    "content": tars_persona_prompt["content"] + f" The user has requested to set your humor to {humor_level}. Acknowledge this change."
                }
                conversation_history.append(new_prompt)
                tts_model.speak(f"Humor setting adjusted to {humor_level}.")
                continue
            except:
                tts_model.speak("I could not process that request.")
                continue

        conversation_history.append({"role": "user", "content": user_input})
        
        response = llm_model.chat(messages=conversation_history)
        print(f"🤖 {ASSISTANT_NAME}: {response}")
        
        tts_model.speak(response)
        conversation_history.append({"role": "assistant", "content": response})