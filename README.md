# VITS2 inference
Lightweight VITS2 text-to-speech inference tool with ONNX support.

## Prerequisites
1. Python >= 3.10 (3.11 recommended)
2. Clone this repository
3. Install espeak and requirements: `pip install -r requirements.txt`
4. Add your ONNX model and config file to the project folder
5. Adjust your symbols in text/symbols.py if needed

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

### Examples

#### 1. Save only (default):
```
python infer_onnx.py --model model_name.onnx --config-path config.json --text "Всем привет!" --lang ru
```

#### 2. Save and play:
```
python infer_onnx.py --model model_name.onnx --config-path config.json --text "Всем привет!" --lang ru --play
```

#### 3. Interactive mode, save only:
```
python infer_onnx.py --model model_name.onnx --config-path config.json --lang ru
```

#### 4. Interactive mode, save and play:
```
python infer_onnx.py --model model_name.onnx --config-path config.json --lang ru --play
```

## Notes
- The output directory will be created automatically if it doesn't exist
- In interactive mode, you can type 'quit' to exit the program
- Audio playback requires a working audio device on your system