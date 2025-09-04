# qDINA

The Quantum Divergent Index Advisor (qDINA) is a quantum-enabled divergent index tuning advisor, which uses quantum machine learning to recommend indexes for a cluster of fully-replicated databases. It is based on [DINA](https://github.com/const-sambird/dina/tree/classical), a divergent design index tuning advsior. qDINA is a research project to investigate methods of quantum acceleration for the index selection problem on replicated databases.

## Installation

### PostgreSQL

qDINA uses PostgreSQL for its databases. The experiments for the research paper were run using PostgreSQL 17 and HypoPG 1.4.1. Other versions should work as long as the HypoPG syntax is the same and it is compatible with the `psycopg` driver, but this cannot be guaranteed.

Create the cluster of Postgres databases and ensure that each database is configured with the requisite permissions for the connecting user. Install [HypoPG](https://hypopg.readthedocs.io/en/rel1_stable/installation.html) and similarly ensure the connecting user has permission to use it.

In the paper, a single database was used per installation of Postgres, each on a different virtual machine. In theory, there is nothing stopping you from using three different databases on a single Postgres install (as a new connection will be opened to each database), but there may be some performance/concurrency penalty.

### qDINA

```bash
$ git clone https://github.com/const-sambird/dina.git
$ cd dina
```

The default `quantum` branch is the correct version of qDINA to install. The `classical` branch is an earlier version and several modifications to the underlying algorithm (to improve performance and better align with the source paper) have been made since then that impact both the classical and quantum versions. To run classical DINA, simply omit the `-q` command line option (explained below).

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

qDINA requires a `replicas.csv` file to list the database replicas to create (simulated) indexes on. The format that is expected for a single connection is

```
id,hostname,port,dbname,user,password,
```

| Field     | Explanation
|-----------|------------------------------------
| id        | A number to identify the database replica (1, 2, ...)
| hostname  | The IP address of the PostgreSQL database
| port      | Which port number to connect to (the default is 5432 but it must be specified)
| user      | The user to connect with. This user must have sufficient privileges on the database to create and drop hypothetical indexes and run EXPLAIN commands
| password  | The password for the user

One line per replica.

## Running

```
$ python learner.py
        -s [SCALE_FACTOR]           what scale factor the database was generated at, usually 10
        -w [MAX_INDEX_WIDTH]        the maximum number of columns allowed in an index. in our experiments,
                                    2 for TPC-H, 1 for TPC-DS. omit for unlimited (note the state space will
                                    grow exponentially in the worst case)
        --workload-factor [WF]      the coefficient for the workload cost reduction in the reward function.
                                    10 in our experiments
        --eps-decay [DECAY]         how quickly the exploration probability decays. if you change the max number
                                    of epochs, you will probably want to change this
        -e [EPOCHS...]              after this many learning episodes, generate an index configuration (using no
                                    exploration, only exploitation). after the maximum number in this list is reached,
                                    the algorithm terminates
        -b [SPACE_BUDGET]           total secondary index space budget on each replica, measured in bytes. after indexes
                                    are added to reach or exceed this number, a learning episode terminates. note these
                                    are predictions because we do not actually create the indexes, just simulate them
                                    (in cost estimation mode)
        -W {tpc-h, tpc-ds}          which workload are we running?
        --training-set [PATH]       if the queries are pregenerated, where can we find them?
        -q                          is this a quantum experiment?
        -n [NUM_QUBITS]             the number of qubits to use, if this is quantum
        --qnn-output {layer, trunc} the QNN's output dimension will not match the number of actions. should we introduce
                                    a classical layer to map observed qubit states to each action, or just ignore (truncate)
                                    observations that don't make sense?
        --batch-size [BATCH_SIZE]   minibatch size (see paper)
        --param-layers [REPS]       the number of repetitions in the ansatz
        --train-fraction [FRAC]     what percentage of the workload should we be using?
        -t [NUM_QUERIES]            how many queries from each template should we use, in a uniform skew case?
        {low_data,drift,recommend}  which type of experiment should we run?
                                    - low_data: use a limited number of queries per instance for training. select
                                        the training queries at the beginning and use them throughout the entire
                                        training run.
                                    - drift: the workload changes over time, and the distribution from which the
                                        queries are drawn starts to skew.
                                    - recommend: don't bother with this nonsense, just use all of the queries in
                                        the workload for training.
```

Example invocations are given below (the commands run in our actual experimental runs):

| Workload | Experiment | Neural network | Command
|----------|------------|----------------|--------
| TPC-H    | Low data   | Classical      | `python learner.py -s 10 -w 2 -c --workload-factor 10 --eps-decay 50 -e 100 -b 5000000000 -W tpc-h --train-fraction [FRACTION] -t [QUERIES] low_data`
| TPC-H    | Low data   | Quantum        | `python learner.py -s 10 -w 2 -c --workload-factor 10 --eps-decay 50 -e 100 -b 5000000000 -W tpc-h -q -n 11 --batch-size 1 --qnn-output layer --train-fraction [FRACTION] -t [QUERIES] low_data`
| TPC-H    | Drift      | Classical      | `python learner.py -s 10 -w 2 -c --workload-factor 10 --eps-decay 50 -e 100 -b 5000000000 -W tpc-h --train-fraction 0.5 -t 5 low_data`
| TPC-H    | Drift      | Quantum        | `python learner.py -s 10 -w 2 -c --workload-factor 10 --eps-decay 50 -e 100 -b 5000000000 -W tpc-h -q -n 11 --batch-size 1 --qnn-output layer --train-fraction 0.5 -t 5 low_data`
| TPC-DS   | Low data   | Classical      | `python learner.py -s 1 -w 1 -c --workload-factor 10 --eps-decay 50 -e 100 -b 500000000 -W tpc-ds --training-set /proj/qdina-PG0/dina-set/ds/train --train-fraction [FRACTION] -t [QUERIES] low_data`
| TPC-DS   | Low data   | Quantum        | `python learner.py -s 1 -w 1 -c --workload-factor 10 --eps-decay 50 -e 100 -b 500000000 -W tpc-ds --training-set /proj/qdina-PG0/dina-set/ds/train -q -n 11 --qnn-output layer --batch-size 1 --train-fraction [FRACTION] -t [QUERIES] low_data`
| TPC-DS   | Drift      | Classical      | `python learner.py -s 1 -w 1 -c --workload-factor 10 --eps-decay 50 -e 100 -b 500000000 -W tpc-ds --training-set /proj/qdina-PG0/dina-set/ds/train --train-fraction 0.5 -t 5 drift`
| TPC-DS   | Drift      | Quantum        | `python learner.py -s 1 -w 1 -c --workload-factor 10 --eps-decay 50 -e 100 -b 500000000 -W tpc-ds --training-set /proj/qdina-PG0/dina-set/ds/train -q -n 11 --qnn-output layer --batch-size 1 --train-fraction 0.5 -t 5 drift`

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