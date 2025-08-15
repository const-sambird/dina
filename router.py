import numpy as np
import psycopg
import time
import re
from database import Replica
from profiling import Profiler
from workload_manager import WorkloadManager

class Router:
    def __init__(self, queries: list[str], templates: list[int], tables: list[str],
                 replicas: list[Replica], candidates: tuple[str], cols_to_table: dict,
                 profiler: Profiler, mode: str, workload_manager: WorkloadManager):
        self.queries = queries
        self.templates = templates
        self.num_templates = len(list(set(templates)))
        self.tables = tables
        self.replicas = replicas
        self.candidates = candidates
        self.cols_to_table = cols_to_table
        self.num_replicas = len(replicas)
        self.profiler = profiler
        self.mode = mode
        self.workload_manager = workload_manager
        
        self.times = np.zeros((self.num_replicas, self.num_templates), dtype=np.float32)
        self.query_costs = np.full(self.num_templates, float('inf'), dtype=np.float32)
        self.replica_costs = np.full(self.num_replicas, float('inf'), dtype=np.float32)
        self.routes = [-1 for _ in queries]

    def _evaluate_cost(self, configurations: list | None):
        try:
            for i_rep, replica in enumerate(self.replicas):
                #print(f'* benchmarking on replica {i_rep + 1} of {len(self.replicas)}')
                # here, we actually want to open a new connection, so as to not interfere with
                # the existing virtual indexes in our main replica connection
                conn = replica.connection()
                with conn.cursor() as cur:
                    indexes_required = 0

                    if configurations is not None:
                        cur.execute('SELECT hypopg_reset();')
                        for config in configurations[i_rep]:
                            table = config[0]
                            columns = config[1]
                            indexes_required += 1
                            #print(f'creating index {indexes_required} : {table}')
                            creation_string = 'CREATE INDEX candidate_index_%d ON %s (%s)' % (indexes_required, table, ', '.join(columns))
                            cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % creation_string)
                    
                    queries = self.workload_manager.workload()
                    templates = self.workload_manager.templates()

                    for idx, query in enumerate(queries):
                        #print(f'estimating query {idx + 1} cost of {len(self.queries)}')
                        for statement in query.split(';'):
                            if 'create view' in statement or 'drop view' in statement:
                                cur.execute(statement)
                            elif 'select' in statement:
                                cur.execute('EXPLAIN (FORMAT JSON) %s' % statement)
                                if after_timing := cur.fetchone()[0][0]['Plan']['Total Cost']:
                                    self.times[i_rep][templates[idx]] += float(after_timing)
                    
                    if configurations is not None:
                        cur.execute('SELECT hypopg_reset();')
                    
                    conn.commit()

        except Exception as err:
            print('got an exception in the database connection')
            print(err)
            conn.rollback()
    
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
                    
                    for idx, query in enumerate(self.queries):
                        #print(f'testing query {idx + 1} of {len(self.queries)}')
                        tic = time.time()
                        cur.execute(query)
                        toc = time.time()

                        self.times[i_rep][self.templates[idx]] += toc - tic
                    
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

        for replica in range(self.num_replicas):
            self.replica_costs[replica] = 0
            for template in range(self.num_templates):
                if self.routes[template] == replica:
                    self.replica_costs[replica] += self.query_costs[template]

        return self.routes
