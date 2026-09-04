# Full pipeline. `make all` reproduces every number in the write-up.
# On the cluster, run the GPU stages via sbatch instead (see ../slurm/).
PY       ?= /home/samiran2/miniconda3/envs/anchors/bin/python
MODEL    ?= allenai/Olmo-3-7B-Think
LABELLER ?= Qwen/Qwen3-8B

.PHONY: all sim screen traces resample filler analyze indist labels categories figures examples report

all: sim screen traces resample filler analyze indist labels categories figures examples report

sim:        ; $(PY) 06_simulate.py --traces 32
screen:     ; $(PY) 00_screen.py    --model $(MODEL)
traces:     ; $(PY) 01_traces.py    --model $(MODEL)
resample:   ; $(PY) 02_resample.py  --model $(MODEL) --arm main
filler:     ; $(PY) 02_resample.py  --model $(MODEL) --arm filler
analyze:    ; $(PY) 03_analyze.py   --model $(MODEL)
labels:     ; $(PY) 04_labels.py    --target-model $(MODEL) --labeller $(LABELLER)
categories: ; $(PY) 09_categories.py --model $(MODEL)
figures:    ; $(PY) 05_figures.py   --model $(MODEL)
examples:   ; $(PY) 07_examples.py  --model $(MODEL)
report:     ; $(PY) 08_report.py    --model $(MODEL)
indist:     ; $(PY) 10_filler_indistribution.py --model $(MODEL)
