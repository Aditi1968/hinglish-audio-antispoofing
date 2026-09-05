from datasets import Audio, load_dataset

print("Loading Kathbath Hindi...")

dataset = load_dataset(
    "ai4bharat/Kathbath",
    "hindi",
    split="train",
    streaming=True,
)

dataset = dataset.cast_column("audio_filepath", Audio(decode=False))

sample = next(iter(dataset))

print("\nFIELDS:")
print(sample.keys())
print("\nSPEAKER ID:")
print(sample["speaker_id"])
print("\nFILE NAME:")
print(sample["fname"])
print("\nTEXT:")
print(sample["text"])
print("\nDURATION:")
print(sample["duration"])
print("\nGENDER:")
print(sample["gender"])
print("\nLANGUAGE:")
print(sample["lang"])
print("\nAUDIO PAYLOAD:")
print(sample["audio_filepath"])
print("\nAUDIO PAYLOAD TYPE:")
print(type(sample["audio_filepath"]))

if isinstance(sample["audio_filepath"], dict):
    print("\nAudio path:")
    print(sample["audio_filepath"].get("path"))
    audio_bytes = sample["audio_filepath"].get("bytes")
    if audio_bytes is not None:
        print("\nAudio bytes available:")
        print(len(audio_bytes), "bytes")
    else:
        print("\nNo embedded bytes; path will be used.")
