# qDINA

The Quantum Divergent Index Advisor (qDINA) is a quantum-enabled divergent index tuning advisor, which uses quantum machine learning to recommend indexes for a cluster of fully-replicated databases. It is based on [DINA](https://github.com/const-sambird/dina/tree/classical), a divergent design index tuning advsior. qDINA is a research project to investigate methods of quantum acceleration for the index selection problem on replicated databases.

## Installation

qDINA is built on Python 3.12.9, though other versions should be compatible. It is recommended to create a venv to install the packages.

```bash
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

To create the queries (if not done so already in a pregenerated set) the prerequisites for the TPC-H and TPC-DS `qgen`/`dsqgen` programs need to be installed:

```bash
$ sudo apt-get install gcc make flex bison byacc git gcc-9
```

Then, download the runkits from the [TPC website](https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp) and create a Makefile by renaming and editing `makefile.suite` to your system specifications. `qgen` will not compile on macOS without changing references from `malloc.h` to `stdlib.h`. (The experimental results for qDINA were run on Ubuntu 24.04, and a Linux environment is recommended for reproducibility).

## Configuration

## Running

## Further options

```
usage: learner.py [-h] [-q] [-n NUM_QUBITS] [-b SPACE_BUDGET] [-s SCALE_FACTOR] [-e NUM_EPOCHS [NUM_EPOCHS ...]] [-w MAX_INDEX_WIDTH] [-m {cost,exe}]
                  [-o NUM_SHOTS] [-g] [-t QUERIES_PER_TEMPLATE] [-W {tpc-h,tpc-ds}] [-c] [--batch-size BATCH_SIZE] [--discount-rate DISCOUNT_RATE]
                  [--eps-start EPS_START] [--eps-end EPS_END] [--eps-decay EPS_DECAY] [--update-rate UPDATE_RATE] [--learning-rate LEARNING_RATE]
                  [--replay-buffer REPLAY_BUFFER] [--hidden-layers HIDDEN_LAYERS [HIDDEN_LAYERS ...]] [--workload-factor WORKLOAD_FACTOR]
                  [--skew-factor SKEW_FACTOR] [--qnn-output {trunc,layer}] [--seed SEED] [--dry-run] [--workload-dir WORKLOAD_DIR]
                  [--template-dir TEMPLATE_DIR] [--save-model] [--load-model] [--param-layers PARAM_LAYERS] [--train-fraction TRAIN_FRACTION]
                  [--training-set TRAINING_SET]
                  {recommend,low_data,drift}

positional arguments:
  {recommend,low_data,drift}
                        what experiment should we run? recommend indexes (normal), low data (limited templates), or workload drift

options:
  -h, --help            show this help message and exit
  -q, --quantum         use quantum neural networks instead of classical ones
  -n NUM_QUBITS, --num-qubits NUM_QUBITS
                        the number of qubits to use in the quantum neural nets
  -b SPACE_BUDGET, --space-budget SPACE_BUDGET
                        the amount of space on each replica that the indexes are allowed to take (in bytes)
  -s SCALE_FACTOR, --scale-factor SCALE_FACTOR
                        TPC-H scale factor
  -e NUM_EPOCHS [NUM_EPOCHS ...], --num-epochs NUM_EPOCHS [NUM_EPOCHS ...]
                        number of learning episodes
  -w MAX_INDEX_WIDTH, --max-index-width MAX_INDEX_WIDTH
                        maximum number of columns that may form an index
  -m {cost,exe}, --benchmark-mode {cost,exe}
                        benchmark execution mode -- 'cost' for the cost estimator, 'exe' for actual execution times
  -o NUM_SHOTS, --num-shots NUM_SHOTS
                        number of samples to take from the quantum neural network
  -g, --generate-queries
                        generate new queries from the templates
  -t QUERIES_PER_TEMPLATE, --queries-per-template QUERIES_PER_TEMPLATE
                        number of queries per template that are in the workload or should be generated
  -W {tpc-h,tpc-ds}, --workload {tpc-h,tpc-ds}
                        the workload to run (TPC-H, TPC-DS)
  -c, --copy-training-set
                        read queries in from the training set
  --batch-size BATCH_SIZE
                        the batch size to feed into the neural network
  --discount-rate DISCOUNT_RATE
                        the discount rate for the reinforcement learner
  --eps-start EPS_START
                        the starting probability of the reinforcement learner exploration rate
  --eps-end EPS_END     the ending probability of the reinforcement learner exploration rate
  --eps-decay EPS_DECAY
                        the rate at which the exploration probability decays
  --update-rate UPDATE_RATE
                        the rate at which the policy nets are updated
  --learning-rate LEARNING_RATE
                        the rate at which the q-learner learns
  --replay-buffer REPLAY_BUFFER
                        the size of the replay buffer
  --hidden-layers HIDDEN_LAYERS [HIDDEN_LAYERS ...]
                        the hidden layers in the neural network, number of neurons (classical only. ignored for quantum)
  --workload-factor WORKLOAD_FACTOR
                        the weight that the workload time should take in the reward function
  --skew-factor SKEW_FACTOR
                        the weight that the workload skew should take in the reward function
  --qnn-output {trunc,layer}
                        how should we map the output probabilities from the QNN to actions? [trunc]ate them to fit or add a classical [layer] (quantum
                        only)
  --seed SEED           the seed for the PRNG used in exploration
  --dry-run             do not enable logging to weights & biases for this run
  --workload-dir WORKLOAD_DIR
                        the directory where the workload .sql files and template assignment .csv are kept
  --template-dir TEMPLATE_DIR
                        the path to the query templates to generate the workload
  --save-model          write the model weights to disk after training is complete
  --load-model          load model weights from disk before training starts
  --param-layers PARAM_LAYERS
                        the number of repetitions of the ansatz setup
  --train-fraction TRAIN_FRACTION
                        what proportion of the workload should be in the training set?
  --training-set TRAINING_SET
                        the location of the training set queries
```