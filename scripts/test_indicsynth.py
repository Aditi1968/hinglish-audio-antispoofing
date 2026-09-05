"""Quick access + schema check for IndicSynth (Hindi) before we pull 5,000 clips."""

from datasets import load_dataset, Audio

print("Loading IndicSynth Hindi...")

dataset = load_dataset(
    "vdivyasharma/IndicSynth",
    "Hindi",
    split="train",
    streaming=True,
)

# Don't decode/resample the audio yet - we only want to inspect the payload.
dataset = dataset.cast_column("audio", Audio(decode=False))

sample = next(iter(dataset))

print("\nFIELDS:")
print(sample.keys())

print("\nGENERATOR:")
print(sample["Generative Model"])

print("\nSOURCE SPEAKER:")
print(sample["Source Speaker_ID"])

print("\nTARGET SPEAKER:")
print(sample["Target Speaker ID"])

print("\nGENDER:")
print(sample["Gender"])

print("\nTRANSCRIPT:")
print(sample["TTS Transcript"])

print("\nAUDIO:")
print(sample["audio"])
