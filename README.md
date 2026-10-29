# AMT distillation experiments

Code for the experiments in my bachelor thesis: a piano transcription model (teacher) and smaller students trained from it.
Models are in `models/` (the original thesis code), everything else is the training and evaluation pipeline.

## Setup

    pip install -r requirements.txt

`librosa` has to be 0.9.2, the mel code in `models/mel.py` calls it with positional arguments.

## Data

MAESTRO v3.0.0 as it is downloaded (the zip, no need to unpack it):

    python prep_maestro.py --zip maestro-v3.0.0.zip --out data/maestro-pre

This writes `train/`, `validation/` and `test/` from the official split. Mel is stored as float16, labels as uint8.
Synthetic clips (optional) are a folder of `.pt` files in the same format, they are only added to the training set.

## Training

teacher

    python train.py --mode teacher --data data/maestro-pre --out runs/teacher --epochs 100 --lr 3e-4

student from the teacher (the settings I ended up with)

    python train.py --mode student --data data/maestro-pre --out runs/student --teacher runs/teacher/best.pth \
        --epochs 30 --lr 2.64e-4 --weights 1 4.39 1.5 5.15 --alpha 0.93 --temp 3.3 --patience 4

the same student without a teacher: add `--no_kd`. Without `--no_mask` the distillation loss only uses the frames where there is an
onset or the teacher gives it more than 5%. Smaller students: `--mode student_pool4`, `student_pool8`.

Training can be continued with `--resume`. The best checkpoint and the threshold are chosen on validation only.

## Evaluation

    python evaluate.py --mode student --ckpt runs/student/best.pth --data data/maestro-pre --out runs/student/final.json

The threshold for peak picking is chosen on validation, the test split is only used once. Onset f1 is counted per pitch
with 1 and 2 frames tolerance (32 and 64 ms).

## Analysis

`analysis/make_cache.py` adds the note attributes to the test files (or builds the same from SMD wav + mid),
`run_inference.py` saves the onset probabilities of a model, then

- `degradation.py` f1 / recall by register, polyphony, velocity and duration, teacher against student
- `error_types.py` octave, fifth and other mistakes
- `mireval_note.py` note level f1 with 50 ms tolerance (mir_eval)
- `onnx_check.py` onnx export, int8 size, speed

## Not included

The generator of the synthetic clips and the notebooks with the first experiments.
