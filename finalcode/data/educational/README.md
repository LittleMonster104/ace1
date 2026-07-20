# Edu PPT High-Quality Image-Text Dataset

Created: 2026-07-02T12:27:22

This folder contains the filtered high-quality subset of the PPT-derived education image semantic alignment dataset.

## Layout

- `images/`: image files, preserving the relative paths used by `image_path`.
- `annotations/images.jsonl`: one record per image, with bilingual captions.
- `annotations/pairs.jsonl`: one record per image-caption pair.
- `annotations/summary.json`: filtering summary and subject distribution.
- `metadata/image_paths.txt`: image paths included in this subset.

## Filtering Criteria

Kept records satisfy all of the following:

- no annotation error
- no `quality_flags`
- exactly three captions
- aspects are `subject`, `knowledge`, and `teaching_visual`
- both Chinese `zh` and English `en` captions are non-empty

## Counts

- images: 160490
- image-text pairs: 481470
- linked images: 160490
- copied images: 0
- existing images skipped: 0
- missing source images: 0

Images were organized with hard links when possible to avoid duplicating storage. They behave like normal files for reading.
