# Module 1 — Memory-based vs. Generator-based Loading: Comparison

## Results

| Metric | Memory-based (Keras) | Generator-based (PyTorch) |
|---|---|---|
| Setup / full load time | 32.10 sec | 0.0158 sec (indexing only) |
| First full pass over data | included above | 2.42 sec (actual image reads) |
| Memory footprint | 281.25 MB (all images in RAM) | Only current batch in RAM |
| Dataset size tested | 6000 images, 64x64 | 6000 images, 64x64 |

## Observations

- The **memory-based** approach loads and decodes every image upfront, which is why it front-loads
  all the cost into a single ~32s step. After that, every subsequent access is effectively free
  (already in RAM as NumPy arrays).
- The **generator-based** approach defers image reading until iteration time (`__getitem__`), so
  "indexing" is near-instant, but each epoch still pays the image-decoding cost — here ~2.4s per
  epoch, repeated every epoch.
- For this specific dataset (small: 6000 images at 64x64), Keras's `load_img` had noticeably higher
  per-image overhead than PyTorch/PIL's direct loading, which is why the memory-based load step was
  slower here — this is more about the two libraries' image-decoding paths than an inherent property
  of "loading everything vs. loading lazily."

## When to prefer which approach

- **Memory-based**: best when the dataset comfortably fits in RAM and you'll iterate over it many
  times (many training epochs) — pay the loading cost once, then train fast with no repeated disk I/O.
- **Generator-based**: best when the dataset is too large to fit in memory, or when you need on-the-fly
  augmentation, since only the current batch is held in memory at any time.

## Conclusion

Both approaches are implemented and validated on the same dataset. Given this dataset's small size,
either would work fine for training; the generator-based approach was chosen going forward for
flexibility (augmentation, and scalability if the dataset grows).