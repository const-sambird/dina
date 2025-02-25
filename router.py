import numpy as np
import psycopg
import time
from profiling import Profiler

class Router:
    def __init__(self, queries, configurations, num_replicas, profiler: Profiler):
        self.queries = queries
        self.configurations = configurations
        self.num_replicas = num_replicas
        self.profiler = profiler
        
        self.times = np.full((num_replicas, len(queries)), float('inf'))
        self.routes = [-1 for _ in queries]

    def _evaluate_one(self, configuration, replica):
        try:
            with psycopg.connect('dbname=tpchdb user=sam') as conn:
                with conn.cursor() as cur:
                    indexes_required = 0
                    
                    for table, columns in configuration.items():
                        indexes_required += 1
                        cur.execute('CREATE INDEX candidate_index_%d ON %s (%s);' % (indexes_required, table, ', '.join(columns)))
                        print('CREATE INDEX candidate_index_%d ON %s (%s);' % (indexes_required, table, ', '.join(columns)))
                    
                    for idx, query in enumerate(self.queries):
                        tic = time.time()
                        cur.execute(query)
                        toc = time.time()

                        self.times[replica][idx] = toc - tic
                    
                    while indexes_required > 0:
                        cur.execute('DROP INDEX candidate_index_%d;' % indexes_required)
                        indexes_required -= 1
        except Exception as err:
            print('got an exception in the database connection')
            print(err)
    
    def evaluate(self):
        self.profiler.time_in('database.route')
        for replica in range(self.num_replicas):
            self._evaluate_one(self.configurations[replica], replica)
        self.profiler.time_out()

        self.routes = np.argmin(self.times, axis=0)

from preprocessor import Preprocessor

if __name__ == '__main__':
    prof = Profiler()
    p = Preprocessor(prof)
    p.preprocess(2000000)
    config = np.asarray([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    r = Router(p.templates, config, 1, prof)
    r.evaluate()
    print(r.routes)
