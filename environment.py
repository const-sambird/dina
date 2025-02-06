import numpy as np
import gymnasium as gym
import psycopg
import re
from random import randrange

from util import extract_table_from_query

class IndexSelectionEnv(gym.Env):
    def __init__(self, replicas, candidates, candidate_sizes, templates, space_budget, alpha, beta, mode = 'cost'):
        '''
        The mode is how DINA evaluates rewards.
        - `cost`: we use PostgreSQL's cost estimator to evaluate the performance of indexes
        - `exe`:  we actually run queries to evaluate how quickly they execute using a given index
        
        `cost` is the default but one must be chosen
        '''
        assert mode == 'cost' or mode == 'exe', 'unknown execution mode!'
        self.replicas = replicas
        self.candidates = candidates
        self.candidate_sizes = candidate_sizes
        self.mode = mode
        self.space_budget = space_budget
        self.alpha = alpha
        self.beta = beta

        self.replica_cache = [0 for i in range(len(replicas))]
        self.spaces_used = [0 for i in range(len(replicas))]

        self.num_replicas = len(replicas)
        self.num_candidates = len(candidates)
        self.templates = templates

        self._state = np.zeros((self.num_replicas, self.num_candidates))
        
        '''
        The observation space is the set of index configurations on each replica.
        
        The set of index configurations is modelled as an r x m binary matrix, where r is the number of
        replicas in the system and m is the number of columns that exist (ie, it is an index candidate).
        An entry `[i, j]` in this space matrix is 1 if the ith replica has column j in its index, and 0
        otherwise.
        '''
        self.observation_space = gym.spaces.MultiBinary([self.num_replicas, self.num_candidates])
        self.action_space = gym.spaces.Discrete(self.num_candidates * self.num_replicas)

        self._compute_baseline()
    
    def _get_obs(self):
        return self._state
    
    def _get_info(self):
        return {
            'mode': self.mode,
            'alpha': self.alpha,
            'beta': self.beta,
            'cache': self.replica_cache,
            'budget': self.space_budget,
            'spaces_used': self.spaces_used
        }
    
    def reset(self, seed, options):
        super().reset(seed=seed)

        self._state = np.zeros((self.num_replicas, self.num_candidates))
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        candidate_to_add = action % self.num_candidates
        replica_to_update = action // self.num_candidates

        required_space = self.candidate_sizes[candidate_to_add]
        if required_space < self.space_budget - self.spaces_used[replica_to_update]:
            self._drop_candidates_to_free(required_space, replica_to_update, candidate_to_add)

        self._state[replica_to_update][candidate_to_add] = 1
        self.spaces_used[replica_to_update] += required_space

        reward = self.reward(self.candidates[candidate_to_add], replica_to_update)
        observation = self._get_obs()
        info = self._get_info()

        terminated = False # ?
        truncated = False

        return observation, reward, terminated, truncated, info

    def reward(self, proposed_configuration, updated_replica):
        benchmark_fn = None
        if self.mode == 'cost':
            benchmark_fn = self._benchmark_index_cost
        else:
            benchmark_fn = self._benchmark_index_exe
        
        total_cost = 0
        replica_costs = [x for x in self.replica_cache]
        for template in self.templates:
            if cost := benchmark_fn(template,
                                    extract_table_from_query(template),
                                    proposed_configuration,
                                    updated_replica):
                total_cost += cost
        replica_costs[updated_replica] = total_cost
        total_cost = sum(replica_costs)
        
        processing_reward = abs(self.baseline - total_cost) / self.baseline
        skew_reward = self._skew_reward(total_cost, replica_costs)

        return (self.alpha * processing_reward) + (self.beta * skew_reward)

    def _skew_reward(self, total_cost, replica_costs):
        num_replicas = len(replica_costs)
        bestcase = total_cost / num_replicas
        skew = 0

        for cost in replica_costs:
            this_skew = abs(cost - bestcase)
            skew += this_skew / bestcase
        
        return 1 / skew

    def _benchmark_index_exe(self, query: str, table: str, cols_to_index: list[str], replica: str) -> float | None:
        '''
        Returns the *actual execution time* of the given query,
        provided that the candidate index described in `cols_to_index`
        is constructed on the table.

        Returns None if it is not possible to benchmark this candidate.
        '''
        with psycopg.connect('dbname=dina user=sam') as conn:
            with conn.cursor() as cur:
                REGEX = '(([0-9]|\\.)+)'
                
                select_table = table
                if '.' in table:
                    select_table = table.split('.')[0]
                
                cur.execute('CREATE INDEX candidate_index ON %s (%s);' % (table, ', '.join(cols_to_index)))
                cur.execute('EXPLAIN ANALYZE %s;' % query)

                if after_timing := re.search(REGEX, cur.fetchall()[-1], re.IGNORECASE):
                    toc = float(after_timing.group(1))
                else:
                    cur.execute('DROP INDEX "%s"."candidate_index";' % select_table)
                    return None

                cur.execute('DROP INDEX "%s"."candidate_index";' % select_table)
                return toc
    
    def _benchmark_index_cost(self, query: str, table: str, cols_to_index: list[str], replica: str) -> float | None:
        '''
        Returns the *estimated execution cost* of the given query,
        as given by PostgreSQL's cost estimation module, provided
        that the candidate index described in `cols_to_index` is
        constructed on the table.

        Returns None if it is not possible to benchmark this candidate.
        '''
        with psycopg.connect('dbname=dina user=sam') as conn:
            with conn.cursor() as cur:
                REGEX = 'cost=([0-9]+\\.[0-9]+)'
                
                select_table = table
                if '.' in table:
                    select_table = table.split('.')[0]

                cur.execute('CREATE INDEX candidate_index ON %s (%s);' % (table, ', '.join(cols_to_index)))
                cur.execute('EXPLAIN %s;' % query)

                if after_timing := re.search(REGEX, cur.fetchone(), re.IGNORECASE):
                    toc = float(after_timing.group(1))
                else:
                    cur.execute('DROP INDEX "%s"."candidate_index";' % select_table)
                    return None

                cur.execute('DROP INDEX "%s"."candidate_index";' % select_table)
                return toc
    
    def _compute_baseline(self):
        benchmark_fn = None
        if self.mode == 'cost':
            benchmark_fn = self._benchmark_index_cost
        else:
            benchmark_fn = self._benchmark_index_exe

        baseline = 0
        for replica in self.replicas:
            for template in self.templates:
                if cost := benchmark_fn(template, extract_table_from_query(template), [], replica):
                    baseline += cost
        
        self.baseline = baseline

    def _compute_space(self, candidates):
        return sum([self.spaces_used[x] for x in candidates])
    
    def _drop_candidates_to_free(self, required_space, replica, target_candidate_idx):
        # there's definitely a better way to do this
        # fun fact this is the subset sum problem !
        space_freed = 0
        dropped = []
        while space_freed < required_space:
            idx_to_drop = randrange(self.num_candidates)
            if idx_to_drop == target_candidate_idx:
                continue

            space_freed += self.candidate_sizes[idx_to_drop]
            dropped.append(idx_to_drop)
        
        for idx in dropped:
            self._state[replica][idx] = 0
        
        self.spaces_used[replica] -= space_freed
