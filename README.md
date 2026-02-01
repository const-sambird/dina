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

> [!TIP]
> The [benchmarking utility](https://github.com/const-sambird/qdina-bench) has further utilities for creating a workload of queries using the TPC-H qgen utility. This is particularly relevant as the paper's results use the same workload for recommending indexes as evaluating results. It is strongly recommended that this is used for reproducing our results, though of course any workload should work.

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

Example invocations are given below (the commands run in our actual experimental runs):

| Experiment    | Neural network | Command
|---------------|----------------|--------
| # replicas    | Classical      | `python learner.py -s 10 -w 1 -c --workload-factor 10 --eps-decay 50 -e 100 -b 5000000000 -W tpc-h --batch-size 16 --qnn-output layer --num-shots 4096 --training-set /proj/qdina-PG0/qdina-1100 --run-name 6rep --param-layers 10 --spsa-iterations 1 --seed 100 recommend`
| # replicas    | Quantum        | `python learner.py -s 10 -w 1 -c --workload-factor 10 --eps-decay 50 -e 100 -b 5000000000 -W tpc-h -q -n 8 --batch-size 16 --qnn-output layer --num-shots 4096 --training-set /proj/qdina-PG0/qdina-1100 --run-name 6rep --param-layers 10 --spsa-iterations 1 --seed 100 recommend`
| # repetitions | Quantum        | `python learner.py -s 10 -w 1 -c --workload-factor 10 --eps-decay 50 -e 100 -b 5000000000 -W tpc-h -q -n 8 --batch-size 16 --qnn-output layer --num-shots 4096 --training-set /proj/qdina-PG0/qdina-1100 --run-name ansatze --param-layers [REPETITIONS] --spsa-iterations 1 --seed 100 recommend`

For all experimental runs, seeds 100 -- 104 were used. Note that the experiments with different numbers of database replicas are configured by modifying `replicas.csv` (instructions above).
