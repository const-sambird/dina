import numpy as np
import psycopg
import time
from database import Replica
from profiling import Profiler

class Router:
    def __init__(self, queries, configurations, tables: list[str], replicas: list[Replica], profiler: Profiler):
        self.queries = queries
        self.configurations = configurations
        self.tables = tables
        self.replicas = replicas
        self.num_replicas = len(replicas)
        self.profiler = profiler
        
        self.times = np.full((self.num_replicas, len(queries)), float('inf'))
        self.routes = [-1 for _ in queries]

    def _evaluate(self, configurations):
        try:
            for i_rep, replica in enumerate(self.replicas):
                with psycopg.connect(replica.connection_string()) as conn:
                    with conn.cursor() as cur:
                        indexes_required = 0

                        for config in configurations[i_rep]:
                            table = config[0]
                            columns = config[1]
                            indexes_required += 1
                            cur.execute('CREATE INDEX candidate_index_%d ON %s (%s);' % (indexes_required, table, ', '.join(columns)))
                        
                        for idx, query in enumerate(self.queries):
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
        self._evaluate(self.configurations)
        self.profiler.time_out()

        self.routes = np.argmin(self.times, axis=0)

        return self.routes
