# Vertex ID

This repository contains an implementation of the YOLO network in pytorch meant to be applied to LArTPC data to perform particle vertex identification.

This repository contains tools for IO, distributed training, saving and restoring, and doing evaluations of test, train, and inference steps.



### How to run

`python bin/exec.py [train, iotest, inference] -f [input file] yolo`

For help on additional commands, run

`python bin/exec.py --help`

or, for commands specifc to a particular stage, run

`python bin/exec.py [train, iotest, inference] --help`

For network options, run

`python bin/exec.py train yolo --help`

