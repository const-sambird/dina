import psycopg
import re

class CostEstimator:
    def __init__(self, workload_matrix, templates, candidates, tables, replicas, space_budget):
        self.workload_matrix = workload_matrix
        self.templates = templates
        self.candidates = candidates
        self.tables = tables
        self.replicas = replicas
        self.space_budget = space_budget
    
    def query_cost_for_index_config_one_replica(self, configuration, queries, replica):
        total_cost = 0
        REGEX = '(([0-9]|\.)+)'

        with psycopg.connect('dbname=dina user=sam') as conn: # FIXME: num_replica conns
            with conn.cursor() as cur:
                for i, element in enumerate(configuration):
                    cur.execute('CREATE INDEX configured_index_%d ON %s (%s);' % (i, element['table'], ', '.join(element['columns'])))
                
                for query in queries:
                    cur.execute('EXPLAIN ANALYZE %s;' % query)
                    if after_timing := re.search(REGEX, cur.fetchall()[-1], re.IGNORECASE):
                        total_cost += float(after_timing.group(1))
                    else:
                        # uh oh!
                        pass
                
                for i in range(len(configuration)):
                    cur.execute('DROP INDEX configured_index_%d;' % i)
        
        return total_cost
    
    def query_cost_for_index_config(self, configuration, queries, replicas):
        total_cost = 0

        for replica in replicas:
            total_cost += self.query_cost_for_index_config_one_replica(configuration, queries, replica)

        return total_cost

    def index_reward(self, existing_config, index, queries):
        cost_without = self.query_cost_for_index_config(existing_config, queries)
        cost_with = self.query_cost_for_index_config([*existing_config, index], queries)

        return (cost_without - cost_with) / cost_without
    
    def workload_skew(self, replica, configuration, queries, num_replicas, total_cost):
        real = self.query_cost_for_index_config_one_replica(configuration, queries, replica)
        bestcase = total_cost / num_replicas

        return abs(real - bestcase) / bestcase
    
    def skew_reward(self, replicas, configuration, queries):
        total_cost = self.query_cost_for_index_config(configuration, queries, replicas)
        reward = 0

        for replica in replicas:
            reward += self.workload_skew(replica, configuration, queries, len(replicas), total_cost)
        
        return 1 / reward
    
    def reward(self, replicas, configuration, index, queries, alpha, beta):
        return (alpha * self.index_reward(configuration, index, queries)) \
             + (beta * self.skew_reward(replicas, [*configuration, index], queries))
    


