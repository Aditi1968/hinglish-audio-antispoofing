from datasets import load_dataset


ds = load_dataset('ai4bharat/Kathbath', 'hindi', split='train[:1]', streaming=False)
print(type(ds))
print(ds.features)
print(next(iter(ds)))
