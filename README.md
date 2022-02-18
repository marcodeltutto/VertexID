# Vertex ID

This repository contains an implementation of the YOLO network in pytorch meant to be applied to LArTPC data to perform particle vertex identification.

This repository contains tools for IO, distributed training, saving and restoring, and doing evaluations of test, train, and inference steps.



### How to run

```bash
python bin/exec.py mode=train run.id="test" \
	data.data_directory="/path/to/data/directory/" \
	data.file="train_file_name" \
	data.aux_file="test_file_name"
```

For a list of all options, run

```bash
python bin/exec.py --help
```
