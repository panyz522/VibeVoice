import argparse
import os
import re
import traceback
from typing import List, Tuple, Union, Dict, Any, Optional, Set
import time
import torch

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.lora_loading import load_lora_assets
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VoiceMapper:
    """Maps speaker names to voice file paths"""

    def __init__(self):
        self.setup_voice_presets()

        # change name according to our preset wav file
        new_dict = {}
        for name, path in self.voice_presets.items():
            if '_' in name:
                name = name.split('_')[0]

            if '-' in name:
                name = name.split('-')[-1]

            new_dict[name] = path
        self.voice_presets.update(new_dict)
        # print(list(self.voice_presets.keys()))

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")

        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        # Scan for all WAV files in the voices directory
        self.voice_presets = {}

        # Get all .wav files in the voices directory
        wav_files = [
            f
            for f in os.listdir(voices_dir)
            if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))
        ]

        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path

        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))

        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items() if os.path.exists(path)
        }

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        if not self.voice_presets:
            raise ValueError("No voice presets available.")

        # First try exact match
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        # Try partial matching (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path

        # Default to first voice if no match found
        default_voice = list(self.voice_presets.values())[0]
        print(
            f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}"
        )
        return default_voice


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    """
    Parse txt script content and extract speakers and their text
    Fixed pattern: Speaker 1, Speaker 2, Speaker 3, Speaker 4
    Returns: (scripts, speaker_numbers)
    """
    lines = txt_content.strip().split('\n')
    scripts = []
    speaker_numbers = []

    # Pattern to match "Speaker X:" format where X is a number
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'

    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            # If we have accumulated text from previous speaker, save it
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)

            # Start new speaker
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            # Continue text for current speaker
            if current_text:
                current_text += " " + line
            else:
                current_text = line

    # Don't forget the last speaker
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)

    return scripts, speaker_numbers


def collect_input_files(
    single_path: Optional[str],
    multiple_paths: Optional[List[str]],
    input_dir: Optional[str],
) -> List[str]:
    """Resolve provided CLI arguments into a list of .txt files."""

    collected: List[str] = []
    seen: Set[str] = set()

    def add_candidate(path: str) -> None:
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            return
        seen.add(abs_path)
        collected.append(abs_path)

    def handle_path(path: Optional[str]) -> None:
        if not path:
            return
        if not os.path.exists(path):
            print(f"Warning: Path not found and will be skipped: {path}")
            return
        if os.path.isdir(path):
            txt_files = [
                os.path.join(path, entry)
                for entry in sorted(os.listdir(path))
                if entry.lower().endswith('.txt') and os.path.isfile(os.path.join(path, entry))
            ]
            if not txt_files:
                print(f"Warning: No .txt files found in directory: {path}")
            for txt_file in txt_files:
                add_candidate(txt_file)
            return
        if not path.lower().endswith('.txt'):
            print(f"Warning: Skipping non-txt file: {path}")
            return
        add_candidate(path)

    handle_path(single_path)

    if multiple_paths:
        for candidate in multiple_paths:
            handle_path(candidate)

    handle_path(input_dir)

    return collected


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Processor TXT Input Test")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-1.5b",
        help="Path to the HuggingFace model directory",
    )

    parser.add_argument(
        "--txt_path",
        type=str,
        default=None,
        help="Path to a txt file containing the script. If this points to a directory, all .txt files inside will be processed.",
    )
    parser.add_argument(
        "--txt_paths",
        type=str,
        nargs='+',
        default=None,
        help="Optional list of txt files to process.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing txt files to process.",
    )
    parser.add_argument(
        "--speaker_names",
        type=str,
        nargs='+',
        default=None,
        help="Speaker names in order (e.g., --speaker_names Andrew Ava 'Bill Gates')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
        help="Device for inference: cuda | mps | cpu",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a fine-tuned checkpoint directory containing LoRA adapters (optional)",
    )
    parser.add_argument(
        "--disable_prefill",
        action="store_true",
        help="Disable speech prefill (voice cloning) by setting is_prefill=False during generation",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.3,
        help="CFG (Classifier-Free Guidance) scale for generation (default: 1.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Normalize potential 'mpx' typo to 'mps'
    if args.device.lower() == "mpx":
        print("Note: device 'mpx' detected, treating it as 'mps'.")
        args.device = "mps"

    # Validate mps availability if requested
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    if args.seed is not None:
        print(f"Setting seed: {args.seed}")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.speaker_names is None:
        args.speaker_names = ["Andrew"]
    elif isinstance(args.speaker_names, str):
        args.speaker_names = [args.speaker_names]

    if not args.txt_path and not args.txt_paths and not args.input_dir:
        args.txt_path = "demo/text_examples/1p_abs.txt"

    input_files = collect_input_files(args.txt_path, args.txt_paths, args.input_dir)

    # sort chunk_{id}.txt inputs numerically, fall back to names for others
    def chunk_id_sort_key(path: str) -> Tuple[int, Union[int, str]]:
        basename = os.path.basename(path)
        match = re.search(r"chunk_(\d+)\.txt$", basename, re.IGNORECASE)
        if match:
            return (0, int(match.group(1)))
        return (1, basename)

    input_files = sorted(input_files, key=chunk_id_sort_key)

    if not input_files:
        print("Error: No valid txt files to process.")
        return

    print(f"Discovered {len(input_files)} txt file(s) to process.")

    voice_mapper = VoiceMapper()
    if not voice_mapper.voice_presets:
        print("Error: No voice presets available. Please add voice samples before running inference.")
        return

    speaker_name_mapping = {str(i): name for i, name in enumerate(args.speaker_names, 1)}

    print("\nConfigured speaker names:")
    for idx, name in speaker_name_mapping.items():
        print(f"  Speaker {idx} -> {name}")

    print(f"\nLoading processor & model from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)

    # Decide dtype & attention implementation
    if args.device == "mps":
        load_dtype = torch.float32  # MPS requires float32
        attn_impl_primary = "sdpa"  # flash_attention_2 not supported on MPS
    elif args.device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
    else:  # cpu
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"

    print(f"Using device: {args.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")

    # Load model with device-specific logic
    try:
        if args.device == "mps":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                attn_implementation=attn_impl_primary,
                device_map=None,  # load then move
            )
            model.to("mps")
        elif args.device == "cuda":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map="cuda",
                attn_implementation=attn_impl_primary,
            )
        else:  # cpu
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map="cpu",
                attn_implementation=attn_impl_primary,
            )
    except Exception as e:
        if attn_impl_primary == 'flash_attention_2':
            print(f"[ERROR] : {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print(
                "Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality."
            )
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map=(args.device if args.device in ("cuda", "cpu") else None),
                attn_implementation='sdpa'
            )
            if args.device == "mps":
                model.to("mps")
        else:
            raise e

    if args.checkpoint_path:
        print(f"Loading fine-tuned assets from {args.checkpoint_path}")
        try:
            report = load_lora_assets(model, args.checkpoint_path)
            loaded_components = [
                name
                for name, loaded in (
                    ("language LoRA", report.language_model),
                    ("diffusion head LoRA", report.diffusion_head_lora),
                    ("diffusion head weights", report.diffusion_head_full),
                    ("acoustic connector", report.acoustic_connector),
                    ("semantic connector", report.semantic_connector),
                )
                if loaded
            ]
            if loaded_components:
                print(f"Loaded components: {', '.join(loaded_components)}")
            else:
                print("Warning: no adapter components were loaded; check the checkpoint path.")
            if report.adapter_root is not None:
                print(f"Adapter assets resolved to: {report.adapter_root}")
        except Exception as exc:
            print(f"Failed to load LoRA assets: {exc}")
            raise

    if args.disable_prefill:
        print("Voice cloning disabled: running generation with is_prefill=False")
    else:
        print("Voice cloning enabled: running generation with is_prefill=True")

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    if hasattr(model.model, 'language_model'):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")

    target_device = args.device if args.device in ("cuda", "mps") else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    for idx, txt_path in enumerate(input_files, 1):
        print("\n" + "-" * 50)
        print(f"Processing file {idx}/{len(input_files)}: {txt_path}")

        if not os.path.exists(txt_path):
            print(f"Warning: txt file not found: {txt_path}")
            continue

        print(f"Reading script from: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()

        scripts, speaker_numbers = parse_txt_script(txt_content)

        if not scripts:
            print("Error: No valid speaker scripts found in the txt file")
            continue

        print(f"Found {len(scripts)} speaker segments:")
        for i, (script, speaker_num) in enumerate(zip(scripts, speaker_numbers)):
            print(f"  {i+1}. Speaker {speaker_num}")
            print(f"     Text preview: {script[:100]}...")

        sorted_unique_speaker_numbers: List[str] = []
        seen_speakers: Set[str] = set()
        for speaker_num in speaker_numbers:
            if speaker_num not in seen_speakers:
                sorted_unique_speaker_numbers.append(speaker_num)
                seen_speakers.add(speaker_num)

        sorted_unique_speaker_numbers.sort(key=lambda x: int(x))
        print("\nSpeaker mapping for this file:")
        for speaker_num in sorted_unique_speaker_numbers:
            mapped_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
            print(f"  Speaker {speaker_num} -> {mapped_name}")

        voice_samples: List[str] = []
        for speaker_num in sorted_unique_speaker_numbers:
            speaker_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
            voice_path = voice_mapper.get_voice_path(speaker_name)
            voice_samples.append(voice_path)
            print(f"Speaker {speaker_num} ('{speaker_name}') -> Voice: {os.path.basename(voice_path)}")

        full_script = '\n'.join(scripts)
        full_script = full_script.replace("’", "'")

        inputs = processor(
            text=[full_script],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)

        print(f"Starting generation with cfg_scale: {args.cfg_scale}")

        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=args.cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
            verbose=True,
            is_prefill=not args.disable_prefill,
        )
        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        audio_duration = None
        rtf = None
        output_path = None

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            txt_filename = os.path.splitext(os.path.basename(txt_path))[0]
            output_path = os.path.join(args.output_dir, f"{txt_filename}_generated.wav")
            processor.save_audio(
                outputs.speech_outputs[0],
                output_path=output_path,
            )
            print(f"Saved output to {output_path}")

            sample_rate = 24000
            audio = outputs.speech_outputs[0]
            audio_samples = audio.shape[-1] if hasattr(audio, "shape") else len(audio)
            audio_duration = audio_samples / sample_rate if audio_samples else 0.0
            rtf = generation_time / audio_duration if audio_duration else float('inf')

            print(f"Generated audio duration: {audio_duration:.2f} seconds")
            print(f"RTF (Real Time Factor): {rtf:.2f}x")
        else:
            print("No audio output generated")

        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = outputs.sequences.shape[1]
        generated_tokens = output_tokens - input_tokens

        print(f"Prefilling tokens: {input_tokens}")
        print(f"Generated tokens: {generated_tokens}")
        print(f"Total tokens: {output_tokens}")

        summary = {
            "input_file": txt_path,
            "output_file": output_path,
            "speaker_count": len(sorted_unique_speaker_numbers),
            "segment_count": len(scripts),
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": output_tokens,
            "generation_time": generation_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
        }
        results.append(summary)

        print("\n" + "=" * 50)
        print(f"GENERATION SUMMARY - {os.path.basename(txt_path)}")
        print("=" * 50)
        print(f"Input file: {summary['input_file']}")
        if summary['output_file']:
            print(f"Output file: {summary['output_file']}")
        else:
            print("Output file: <not generated>")
        print(f"Speaker names: {args.speaker_names}")
        print(f"Number of unique speakers: {summary['speaker_count']}")
        print(f"Number of segments: {summary['segment_count']}")
        print(f"Prefilling tokens: {summary['input_tokens']}")
        print(f"Generated tokens: {summary['generated_tokens']}")
        print(f"Total tokens: {summary['total_tokens']}")
        print(f"Generation time: {summary['generation_time']:.2f} seconds")
        if summary['audio_duration'] is not None:
            print(f"Audio duration: {summary['audio_duration']:.2f} seconds")
        if summary['rtf'] is not None:
            print(f"RTF (Real Time Factor): {summary['rtf']:.2f}x")
        if args.seed is not None:
            print(f"Seed used: {args.seed}")
        print("=" * 50)

    if len(results) > 1:
        total_gen_time = sum(item["generation_time"] for item in results)
        total_audio_duration = sum(
            item["audio_duration"] for item in results if item["audio_duration"] is not None
        )

        print("\n" + "=" * 50)
        print("BATCH SUMMARY")
        print("=" * 50)
        print(f"Files processed: {len(results)}")
        print(f"Outputs saved in: {os.path.abspath(args.output_dir)}")
        print(f"Total generation time: {total_gen_time:.2f} seconds")
        if total_audio_duration:
            print(f"Total audio duration: {total_audio_duration:.2f} seconds")
        print("=" * 50)


if __name__ == "__main__":
    main()
