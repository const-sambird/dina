import numpy as np
import psycopg
import time
import re
from multiprocessing import Process, Queue
from cost_estimator import CostEstimator
from database import Replica
from profiling import Profiler
from workload_manager import WorkloadManager

class Router:
    def __init__(self, queries: list[str], templates: list[int], tables: list[str],
                 replicas: list[Replica], candidates: tuple[str], cols_to_table: dict,
                 profiler: Profiler, mode: str, workload_manager: WorkloadManager):
        self.tables = tables
        self.replicas = replicas
        self.candidates = candidates
        self.cols_to_table = cols_to_table
        self.num_replicas = len(replicas)
        self.profiler = profiler
        self.mode = mode
        self.workload_manager = workload_manager
        self.num_templates = workload_manager.num_full_templates()
        
        self.times = np.zeros((self.num_replicas, self.num_templates), dtype=np.float32)
        self.query_costs = np.zeros(self.num_templates, dtype=np.float32)
        self.replica_costs = np.zeros(self.num_replicas, dtype=np.float32)
        self.routes = [-1 for _ in range(self.num_templates)]

        if mode == 'cost':
            self.cost_queues = [Queue() for _ in replicas]
            self.cost_estimators = [CostEstimator(workload_manager.num_full_templates(), replicas[i].connection_string(), self.cost_queues[i])
                                    for i in range(self.num_replicas)]

    def _evaluate_cost(self, configurations: list | None):
        #try:
        processes = []
        for i_rep, estimator in enumerate(self.cost_estimators):
            processes.append(Process(
                target=estimator.run,
                args=(
                    self.workload_manager.workload(),
                    self.workload_manager.templates(),
                    [] if configurations is None else configurations[i_rep]
                )
            ))
        [p.start() for p in processes]
        [p.join() for p in processes]
        self.times = [q.get() for q in self.cost_queues]

        #except Exception as err:
        #    print('got an exception in the database connection')
        #    print(err)
    
    def _evaluate_exe(self, configurations):
        try:
            for i_rep, replica in enumerate(self.replicas):
                #print(f'* benchmarking on replica {i_rep + 1} of {len(self.replicas)}')
                conn = replica.connection()
                with conn.cursor() as cur:
                    indexes_required = 0

                    if configurations is not None:
                        for config in configurations[i_rep]:
                            table = config[0]
                            columns = config[1]
                            indexes_required += 1
                            #print(f'creating index {indexes_required} : {table}')
                            cur.execute('CREATE INDEX candidate_index_%d ON %s (%s);' % (indexes_required, table, ', '.join(columns)))
                    
                    queries = self.workload_manager.workload()
                    templates = self.workload_manager.templates()
                    
                    for idx, query in enumerate(queries):
                        #print(f'testing query {idx + 1} of {len(self.queries)}')
                        tic = time.time()
                        cur.execute(query)
                        toc = time.time()

                        self.times[i_rep][templates[idx]] += toc - tic
                    
                    if configurations is not None:
                        while indexes_required > 0:
                            cur.execute('DROP INDEX candidate_index_%d;' % indexes_required)
                            indexes_required -= 1
                    
                    conn.commit()

        except Exception as err:
            print('got an exception in the database connection')
            print(err)
            conn.rollback()
    
    def parse_state_matrix(self, state):
        parsed_config = []

        for idx in range(self.num_replicas):
            indexes = []
            for can_idx, include in enumerate(state[idx]):
                if include == 1:
                    indexes.append(can_idx)
            indexes = [self.candidates[can_idx] for can_idx in indexes]
            # add the table name too
            indexes = [[self.cols_to_table[x[0]], x] for x in indexes]
            parsed_config.append(indexes)
        
        return parsed_config
    
    def evaluate(self, configurations: np.ndarray | None = None):
        '''
        Evaluates the cost of every query across the various replicas,
        and updates the routeing table and the minimum cost of processing
        each query.

        :param configurations:  the current state array, or `None` if the
                                information is already persisted in the
                                connection object.
        '''
        self.times = np.zeros((self.num_replicas, self.num_templates), dtype=np.float32)
        if configurations is not None:
            configurations = self.parse_state_matrix(configurations)
        for replica in self.replicas:
            replica.drop_all_indexes(self.tables, 'exe')
        self.profiler.time_in('database.route')
        if self.mode == 'cost':
            self._evaluate_cost(configurations)
        else:
            self._evaluate_exe(configurations)
        self.profiler.time_out()

        self.routes = np.argmin(self.times, axis=0)
        self.query_costs = [self.times[rep][i] for i, rep in enumerate(self.routes)]

        # if this template is in the update workload, we need to route it to every replica
        update_templates = self.workload_manager.update_templates()
        for template in range(self.num_templates):
            if template in update_templates:
                self.routes[template] = -1

        for replica in range(self.num_replicas):
            self.replica_costs[replica] = 0
            for template in range(self.num_templates):
                if self.routes[template] == replica or self.routes[template] == -1:
                    self.replica_costs[replica] += self.query_costs[template]
        
        # if a template isn't present in the training set (it has a cost of zero)
        # assign it to the least-loaded replica
        min_replica = np.argmin(self.replica_costs)
        for i, q_cost in enumerate(self.query_costs):
            if q_cost == 0:
                self.routes[i] = min_replica

        return self.routes
