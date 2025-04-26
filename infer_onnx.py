import numpy as np
import onnxruntime
import torch
from scipy.io.wavfile import write
import time
import soundfile as sf
import sounddevice as sd
import argparse
import os

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
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()

def parse_arguments():
    parser = argparse.ArgumentParser(description="VITS2 Text-to-Speech Inference")
    parser.add_argument("--model", type=str, default="model_name.onnx", 
                       help="Path to the ONNX model")
    parser.add_argument("--config-path", type=str, default="config.json", 
                       help="Path to the configuration JSON file")
    parser.add_argument("--output-wav-path", type=str, default=None,
                       help="Path to save the output WAV file")
    parser.add_argument("--text", type=str, default=None,
                       help="Text to synthesize (if not provided, interactive mode will be started)")
    parser.add_argument("--lang", type=str, default="en",
                       help="Language code (default: en)")
    parser.add_argument("--sid", type=int, default=None,
                       help="Speaker ID for multi-speaker models (default: None)")
    parser.add_argument("--play", action="store_true",
                       help="Play audio after synthesis (default: False)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    model_path = args.model
    config_path = args.config_path
    output_dir = "output" if args.output_wav_path is None else os.path.dirname(args.output_wav_path)
    lang = args.lang
    sid = args.sid
    should_play = args.play
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])

    hps = utils.get_hparams_from_file(config_path)
    
    # If text is provided as argument, synthesize it and exit
    if args.text:
        if args.output_wav_path:
            output_path = args.output_wav_path
        else:
            output_path = f"{output_dir}/output.wav"
        
        generation_time = synthesize_speech(args.text, model, hps, output_path, lang, sid)
        
        print(f"Speech synthesized and saved to {output_path}")
        print(f"Generation time: {generation_time:.2f} seconds")
        
        if should_play:
            print("Playing audio...")
            play_audio(output_path)
        return

    # If no text provided, enter interactive mode
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
        
        if should_play:
            print("Playing audio...")
            play_audio(output_path)
        
        counter += 1

    print("Thank you for using the interactive TTS system!")

if __name__ == "__main__":
    main()