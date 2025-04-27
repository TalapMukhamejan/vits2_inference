# VITS2 inference
I took the code from this [repo](https://github.com/p0p4k/vits2_pytorch) to do inference without any additional files

## Prerequisites
1. Python >= 3.10 (I used 3.11)
2. Clone this repository
3. Install espeak and requirements: `pip install -r requirements.txt`
4. Add your onnx model into folder, with your config file
5. Adjust your symbols in text/symbols.py if needed
6. For Russian accent support: `pip install ruaccent`

## Usage

### Command-line Arguments
```
python infer_onnx.py --model <model_path> --config-path <config_path> [options]
```

### Required Arguments
- `--model`: Path to the ONNX model file
- `--config-path`: Path to the configuration JSON file

### Optional Arguments
- `--output-wav-path`: Path to save the output WAV file (default: output/output.wav)
- `--text`: Text to synthesize (if not provided, interactive mode will be started)
- `--lang`: Language code (default: en)
- `--sid`: Speaker ID for multi-speaker models (default: None)
- `--play`: Play audio after synthesis (default: False)
- `--accent`: Apply Russian accent marks (only works with --lang ru)
- `--accent-model`: RUAccent model size (default: turbo2, options: tiny, tiny2, tiny2.1, turbo2, turbo3, turbo3.1, turbo, big_poetry)
- `--accent-dictionary`: Use dictionary for accents (default: True)
- `--accent-device`: Device to run ruaccent model on (default: CPU, options: CPU, CUDA)

### Examples

#### 1. Save only (default):
```bash
python infer_onnx.py --model model_name.onnx --config-path config.json --text "Всем привет!" --lang ru
```

#### 2. Save and play:
```bash
python infer_onnx.py --model model_name.onnx --config-path config.json --text "Всем привет!" --lang ru --play
```

#### 3. Interactive mode, save only:
```bash
python infer_onnx.py --model model_name.onnx --config-path config.json --lang ru
```

#### 4. Interactive mode, save and play:
```bash
python infer_onnx.py --model model_name.onnx --config-path config.json --lang ru --play
```

#### 5. With Russian accent marks:
```bash
python infer_onnx.py --model model_name.onnx --config-path config.json --text "Я говорю по-русски. Это замок на двери." --lang ru --accent
```


## Russian Accent Support
For better Russian speech synthesis, this tool supports automatic accent placement using the [ruaccent](https://github.com/Den4ikAI/ruaccent) library. This is important because Russian word stress is not fixed and can significantly affect pronunciation.

To use this feature:
1. Install the ruaccent library: `pip install ruaccent`
2. Add the `--accent` flag when using Russian language (`--lang ru`)
3. Optionally customize the accent model with additional parameters:
   - `--accent-model`: Choose different model sizes (smaller = faster, larger = more accurate)
   - `--accent-dictionary`: Enable/disable dictionary usage
   - `--accent-device`: Run on CPU or CUDA (requires onnxruntime-gpu)

## Notes
- The output directory will be created automatically if it doesn't exist
- In interactive mode, you can type 'quit' to exit the program
- Audio playback requires a working audio device on your system
- For GPU acceleration of ruaccent, install onnxruntime-gpu: `pip install onnxruntime-gpu`