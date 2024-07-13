import os
import numpy as np
import onnxruntime
import torch
import soundfile as sf
import time
import winsound

import commons
import utils
from text import text_to_sequence

def get_text(text, hps, lang):
    text_norm = text_to_sequence(text, hps.data.text_cleaners, lang)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def synthesize_speech(text, model, hps, output_path, lang, sid):
    start_time = time.time()
    
    phoneme_ids = get_text(text, hps, lang)
    text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    text_lengths = np.array([text.shape[1]], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    sid = np.array([int(sid)]) if sid is not None else None

    audio = model.run(
        None,
        {
            "input": text,
            "input_lengths": text_lengths,
            "scales": scales,
            "sid": sid,
        },
    )[0].squeeze((0, 1))

    sf.write(output_path, audio, hps.data.sampling_rate)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    return generation_time

def play_audio(file_path):
    winsound.PlaySound(file_path, winsound.SND_FILENAME)

def main():
    # Set these variables manually before running the script
    model_path = "model_name.onnx"
    config_path = "config.json"  # Updated config filename
    output_dir = "output"
    lang = "en"
    sid = None  # Set to an integer if needed

    # Print current working directory and list files
    print(f"Current working directory: {os.getcwd()}")
    print("Files in current directory:")
    for file in os.listdir():
        print(f"  - {file}")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please make sure the ONNX model file is in the same directory as this script.")
        return

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        print("Please make sure the config JSON file is in the same directory as this script.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        sess_options = onnxruntime.SessionOptions()
        model = onnxruntime.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])

        hps = utils.get_hparams_from_file(config_path)

        print("Interactive Text-to-Speech")
        print("Enter 'quit' to exit the program")

        counter = 1
        while True:
            text = input("Enter text to synthesize: ")
            
            if text.lower() == 'quit':
                break

            output_path = f"{output_dir}/output_{counter}.wav"
            
            generation_time = synthesize_speech(text, model, hps, output_path, lang, sid)
            
            print(f"Speech synthesized and saved to {output_path}")
            print(f"Generation time: {generation_time:.2f} seconds")
            print("Playing audio...")
            
            play_audio(output_path)
            
            counter += 1

        print("Thank you for using the interactive TTS system!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your model and config files, and ensure all required libraries are installed.")

if __name__ == "__main__":
    main()