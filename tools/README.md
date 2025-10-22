# Error Evaluate v2.2.1 (on Linux)

```sh
uv sync --group eval
TF_ENABLE_ONEDNN_OPTS=0 PYTHONWARNINGS="ignore" python eval.py
TF_ENABLE_ONEDNN_OPTS=0 PYTHONWARNINGS="ignore" python eval_pulse.py
```

## White noise Median Absolute Error (MdAE)

## Pink noise MdAE

## Pulse noise MdAE
