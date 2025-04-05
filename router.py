import numpy as np
import psycopg
import time
import re
from database import Replica
from profiling import Profiler

class Router:
    def __init__(self, queries, configurations, tables: list[str], replicas: list[Replica], profiler: Profiler, mode: str):
        self.queries = queries
        self.configurations = configurations
        self.tables = tables
        self.replicas = replicas
        self.num_replicas = len(replicas)
        self.profiler = profiler
        self.mode = mode
        
        self.times = np.full((self.num_replicas, len(queries)), float('inf'))
        self.routes = [-1 for _ in queries]

    def _evaluate_cost(self, configurations):
        try:
            for i_rep, replica in enumerate(self.replicas):
                print(f'* benchmarking on replica {i_rep + 1} of {len(self.replicas)}')
                with psycopg.connect(replica.connection_string()) as conn:
                    with conn.cursor() as cur:
                        indexes_required = 0

                        for config in configurations[i_rep]:
                            table = config[0]
                            columns = config[1]
                            indexes_required += 1
                            print(f'creating index {indexes_required} : {table} ({', '.join(columns)})')
                            creation_string = 'CREATE INDEX candidate_index_%d ON %s (%s);' % (indexes_required, table, ', '.join(columns))
                            cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % creation_string)
                        
                        REGEX = 'cost=([0-9]+\\.[0-9]+)'

                        for idx, query in enumerate(self.queries):
                            print(f'estimating query {idx + 1} cost of {len(self.queries)}')
                            cur.execute('EXPLAIN %s;' % query)
                            if after_timing := re.search(REGEX, cur.fetchone()[0], re.IGNORECASE):
                                self.times[i_rep][idx] = float(after_timing.group(1))
                        
                        cur.execute('SELECT hypopg_reset();')

        except Exception as err:
            print('got an exception in the database connection')
            print(err)
    
    def _evaluate_exe(self, configurations):
        try:
            for i_rep, replica in enumerate(self.replicas):
                print(f'* benchmarking on replica {i_rep + 1} of {len(self.replicas)}')
                with psycopg.connect(replica.connection_string()) as conn:
                    with conn.cursor() as cur:
                        indexes_required = 0

                        for config in configurations[i_rep]:
                            table = config[0]
                            columns = config[1]
                            indexes_required += 1
                            print(f'creating index {indexes_required} : {table} ({', '.join(columns)})')
                            cur.execute('CREATE INDEX candidate_index_%d ON %s (%s);' % (indexes_required, table, ', '.join(columns)))
                        
                        for idx, query in enumerate(self.queries):
                            print(f'testing query {idx + 1} of {len(self.queries)}')
                            tic = time.time()
                            cur.execute(query)
                            toc = time.time()

                            self.times[i_rep][idx] = toc - tic
                        
                        while indexes_required > 0:
                            cur.execute('DROP INDEX candidate_index_%d;' % indexes_required)
                            indexes_required -= 1

        except Exception as err:
            print('got an exception in the database connection')
            print(err)
    
    def evaluate(self):
        for replica in self.replicas:
            replica.drop_all_indexes(self.tables, 'exe')
        self.profiler.time_in('database.route')
        if self.mode == 'cost':
            self._evaluate_cost(self.configurations)
        else:
            self._evaluate_exe(self.configurations)
        self.profiler.time_out()

        self.routes = np.argmin(self.times, axis=0)

        return self.routes
