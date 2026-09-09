# Error Evaluate (on CPU only Linux)

```sh
uv sync --group eval
cd tools/v2

TF_ENABLE_ONEDNN_OPTS=0 PYTHONWARNINGS="ignore" python eval.py
TF_ENABLE_ONEDNN_OPTS=0 PYTHONWARNINGS="ignore" python eval_pulse.py
python eval_margin_sensitivity.py
```
