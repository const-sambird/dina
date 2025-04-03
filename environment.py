import numpy as np
import gymnasium as gym
import psycopg
import re
import time
from random import choice

from util import extract_columns_from_query, insert_dummy_values, construct_indexes_from_candidate
from profiling import Profiler
from database import Replica

# a postgres constant. if the space used is within this amount of the 
# space budget, we can never add more indexes
SMALLEST_POSSIBLE_INDEX_SIZE = 16384

class IndexSelectionEnv(gym.Env):
    def __init__(self, profiler: Profiler, replicas: list[Replica], candidates, tables, cols_to_table, templates, queries, space_budget, alpha, beta, mode = 'cost'):
        '''
        The mode is how DINA evaluates rewards.
        - `cost`: we use PostgreSQL's cost estimator to evaluate the performance of indexes
        - `exe`:  we actually run queries to evaluate how quickly they execute using a given index
        
        `cost` is the default but one must be chosen
        '''
        assert mode == 'cost' or mode == 'exe', 'unknown execution mode!'
        assert len(candidates) > 0, 'no candidate indexes! is the space budget prohibitively small?'

        self.profiler = profiler
        self.replicas = replicas
        self.candidates = candidates
        self.cols_to_table = cols_to_table
        self.mode = mode
        self.space_budget = space_budget
        self.alpha = alpha
        self.beta = beta

        self.replica_cache = [0 for i in range(len(replicas))]
        self.spaces_used = [0 for i in range(len(replicas))]
        self.candidate_sizes = {}

        self.num_replicas = len(replicas)
        self.num_candidates = len(candidates)
        self.templates = templates
        self.queries = queries
        self.tables = tables

        self._action_mask = np.ones(shape=(self.num_replicas * self.num_candidates,), dtype=np.int8)

        '''
        The HypoPG what-if optimiser returns oids that represent the virtual indexes. We need to
        store these oids so that we can drop them when the action passed to step() calls for them
        to be removed. (The alternative is dropping all indexes and recreating them, which wouldn't
        be ideal, so we will avoid that if possible).
        '''
        self._virtual_index_oids = np.zeros((self.num_replicas, self.num_candidates), dtype=np.uint32)

        self._state = np.zeros((self.num_replicas, self.num_candidates))
        
        '''
        The observation space is the set of index configurations on each replica.
        
        The set of index configurations is modelled as an r x m binary matrix, where r is the number of
        replicas in the system and m is the number of columns that exist (ie, it is an index candidate).
        An entry `[i, j]` in this space matrix is 1 if the ith replica has column j in its index, and 0
        otherwise.
        '''
        self.observation_space = gym.spaces.MultiBinary([self.num_replicas, self.num_candidates])
        
        '''
        The action space is the set of index configurations on each replica, but each candidate on
        each replica may either be created or dropped.
        '''
        self.action_space = gym.spaces.Discrete(self.num_candidates * self.num_replicas)

        self._drop_all_indexes('cost')
        self._drop_all_indexes('exe')
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
            'spaces_used': self.spaces_used,
            'mask': self._action_mask
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._state = np.zeros((self.num_replicas, self.num_candidates))
        self.spaces_used = [0 for i in range(self.num_replicas)]
        self.replica_cache = [0 for i in range(self.num_replicas)]
        self._action_mask = np.ones((self.num_replicas * self.num_candidates,), dtype=np.int8)
        self._virtual_index_oids = np.zeros((self.num_replicas, self.num_candidates), dtype=np.uint32)
        observation = self._get_obs()
        info = self._get_info()

        self.profiler.count = 0

        self._drop_all_indexes(self.mode)

        return observation, info
    
    def step(self, action: int):
        '''
        `step` is called whenever one action is performed by the reinforcement learning
        algorithm. This action operates on the current state, and returns a tuple representing
        the reward computed for this step, along with the next state.

        `action` is an integer, and represents an index into the state space, multiplied by
        2. If `action` is less than half the size of the action space, we are dropping the
        index; if more than half, we are creating the index. The state space
        is a (num_replicas, num_candidates) array, so the action represents the
        candidate in a given replica that we would like to toggle (ie, add to the index
        configuration if we are in that half of the action space, or drop otherwise).

        We return a tuple:
        - `observation`, the updated state after this action completes
        - `reward`, a float representing the value of this action
        - `terminated`, whether this learning epoch should be terminated
        - `truncated`, the functionality of which I'm honestly not sure about (`False`)
        - `info`, more information about the environment's state (see `IndexSelectionEnv#_get_info`)
        '''
        print(f'* epoch {self.profiler.count}')
        print('action:', action)
        self.profiler.count_up()
        #self.profiler.time_in('step')
        candidate_to_toggle = action % self.num_candidates
        replica_to_update = action // self.num_candidates
        creating = self._state[replica_to_update][candidate_to_toggle] == 0

        if creating:
            self.profiler.time_in('step.compute_size')
            if self._state[replica_to_update][candidate_to_toggle] != 0:
                self.profiler.time_out()
                return self._step_early_continuation(reward=-500)
            print(f'adding {self.candidates[candidate_to_toggle]} on replica {replica_to_update}')
            required_space = self._get_candidate_size(self.candidates[candidate_to_toggle])
            available_space = self.space_budget - self.spaces_used[replica_to_update]
            #if required_space > available_space:
            #    self.profiler.time_out()
            #    return self._step_early_continuation(reward=-1000)

            self.profiler.time_out()
            self.profiler.time_in('step.construct_added_index')
            self._state[replica_to_update][candidate_to_toggle] = 1
            self.spaces_used[replica_to_update] += required_space
            self._construct_index(candidate_to_toggle, replica_to_update)

            if self.space_budget < self.spaces_used[replica_to_update]:
                self._update_mask(replica_to_update)
        else:
            self.profiler.time_in('step.compute_size')
            if self._state[replica_to_update][candidate_to_toggle] != 1:
                self.profiler.time_out()
                return self._step_early_continuation(reward=-500)
            print(f'removing {self.candidates[candidate_to_toggle]} on replica {replica_to_update}')
            required_space = self._get_candidate_size(self.candidates[candidate_to_toggle])
            self._state[replica_to_update][candidate_to_toggle] = 0
            self.spaces_used[replica_to_update] -= required_space

            self.profiler.time_out()
            self.profiler.time_in('step.drop_index')
            self._drop_index(candidate_to_toggle, replica_to_update)

        self.profiler.time_out()
        self.profiler.time_in('step.reward')
        reward = self.reward(replica_to_update)
        self.profiler.time_out()
        truncated = False

        self.profiler.time_in('step.budget_check')
        # are all space budgets full? if so, terminate
        space_budgets_are_full = [1 if i > self.space_budget else 0 for i in self.spaces_used]
        terminated = sum(space_budgets_are_full) == self.num_replicas

        observation = self._get_obs()
        info = self._get_info()

        print(f'spaces used after this epoch: {self.spaces_used} / {self.space_budget}')

        self.profiler.time_out()

        return observation, reward, terminated, truncated, info
    
    def _step_early_termination(self, reward = 0.0):
        '''
        Called when we determine that there is no possible action that we
        can now take that would not exceed the space budget (ie, all replicas
        are full), and we should now terminate this training epoch.

        Equivalent to a `break` statement in the training loop.
        '''
        observation = self._get_obs()
        info = self._get_info()
        terminated = True
        truncated = False
        return observation, reward, terminated, truncated, info
    
    def _step_early_continuation(self, reward = 0.0):
        '''
        Called when an action is passed to the environment that would
        cause us to exceed our space budget if executed, but there do
        still exist some actions that are valid (so we should not terminate
        this training episode).

        Equivalent to a `continue` statement in the training loop.
        '''
        observation = self._get_obs()
        info = self._get_info()
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, info
    
    def _get_candidate_size(self, candidate: tuple[str]) -> int:
        '''
        Determining whether or not we can add this candidate to the state
        requires us to know the size of the candidate. If this is our first
        time encountering this candidate, we will compute its size and cache
        it for later. But if we can return a value from the cache, we will
        do so.
        '''
        if candidate in self.candidate_sizes:
            return self.candidate_sizes[candidate]
        
        computed_size = 0

        try:
            with psycopg.connect(self.replicas[0].connection_string()) as conn:
                with conn.cursor() as cur:
                    # all of the columns in the candidate should be in the same table
                    # so we can pick the first one and find which table it's in
                    table = self.cols_to_table[candidate[0]]
                    creation_string = 'CREATE INDEX candidate_index ON %s (%s);' % (table, ', '.join(candidate))
                    if self.mode == 'exe':
                        cur.execute(creation_string)
                        cur.execute("SELECT pg_table_size('candidate_index');")
                        computed_size = cur.fetchone()[0]
                        cur.execute('DROP INDEX candidate_index;')
                    else:
                        cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % creation_string)
                        virtual_oid = cur.fetchone()[0]
                        cur.execute('SELECT hypopg_relation_size(%s) FROM hypopg_list_indexes;' % virtual_oid)
                        computed_size = cur.fetchone()[0]
                        cur.execute('SELECT hypopg_drop_index(%s);' % virtual_oid)

        except Exception as err:
            print('got an exception in the database connection')
            print(err)

        self.candidate_sizes[candidate] = computed_size
        return computed_size

    def reward(self, updated_replica):
        benchmark_fn = None
        if self.mode == 'cost':
            benchmark_fn = self._benchmark_index_cost
        else:
            benchmark_fn = self._benchmark_index_exe
        
        total_cost = 0
        replica_costs = [x for x in self.replica_cache]
        
        self.profiler.time_out()
        self.profiler.time_in('database.benchmark')
        total_cost = benchmark_fn(self.queries, self.replicas[updated_replica])
        self.profiler.time_out()
        self.profiler.time_in('step')

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
        
        if skew == 0:
            return 1000
        return 1 / skew

    def _benchmark_index_exe(self, queries: list[str], replica: Replica) -> float | None:
        '''
        Returns the *actual execution time* of the given queries,
        provided that the candidate index described in `cols_to_index`
        is constructed on the table(s).

        Returns None if it is not possible to benchmark this candidate.
        '''
        try:
            with psycopg.connect(replica.connection_string()) as conn:
                with conn.cursor() as cur:
                    tic = time.time()

                    for query in queries:
                        cur.execute('%s;' % query)

                    toc = time.time()

                    return toc - tic
        except Exception as err:
            print('got an exception in the database connection')
            print(err)
            return 0

    def _benchmark_index_cost(self, queries: list[str], replica: Replica) -> float | None:
        '''
        Returns the *estimated execution cost* of the given queries,
        as given by PostgreSQL's cost estimation module, provided
        that the candidate index described in `cols_to_index` is
        constructed on the table(s).

        Returns None if it is not possible to benchmark this candidate.
        '''
        try:
            with psycopg.connect(replica.connection_string()) as conn:
                with conn.cursor() as cur:
                    REGEX = 'cost=([0-9]+\\.[0-9]+)'
                    cost = 0

                    for query in queries:
                        cur.execute('EXPLAIN %s;' % query)
                        if after_timing := re.search(REGEX, cur.fetchone()[0], re.IGNORECASE):
                            cost += float(after_timing.group(1))
                    
                    return cost
        except Exception as err:
            print('got an exception in the database connection')
            print(err)
            return 0
        
    def _construct_index(self, candidate_index: int, replica_index: int):
        '''
        Constructs one index candidate on the given replica.
        Throws if the index candidate already exists (this should not happen!)
        '''
        replica = self.replicas[replica_index]
        try:
            with psycopg.connect(replica.connection_string()) as conn:
                with conn.cursor() as cur:
                    candidate = self.candidates[candidate_index]
                    table = self.cols_to_table[candidate[0]]
                    creation_string = 'CREATE INDEX candidate_index_%d ON %s (%s);' % (candidate_index, table, ', '.join(candidate))
                    if self.mode == 'cost':
                        cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % creation_string)
                        self._virtual_index_oids[replica_index][candidate_index] = cur.fetchone()[0]
                    else:
                        cur.execute(creation_string)
        except Exception as err:
            print(f'got an exception in the database connection while constructing index {candidate_index} on replica {replica.id}')
            print(err)
    
    def _drop_index(self, candidate_index: int, replica_index: Replica):
        '''
        Drops one index candidate from the given replica.
        '''
        replica = self.replicas[replica_index]
        try:
            with psycopg.connect(replica.connection_string()) as conn:
                with conn.cursor() as cur:
                    if self.mode == 'cost':
                        virtual_oid = self._virtual_index_oids[replica_index][candidate_index]
                        if virtual_oid == 0:
                            print('********* missing oid for virtual index %d on replica %d !!' % (candidate_index, replica_index))
                        cur.execute('SELECT hypopg_drop_index(%s);' % self._virtual_index_oids[replica_index][candidate_index])
                    else:
                        cur.execute('DROP INDEX candidate_index_%d;' % candidate_index)
        except Exception as err:
            print(f'got an exception in the database connection while dropping index {candidate_index} on replica {replica.id}')
            print(err)
    
    def _drop_all_indexes(self, mode: str = 'cost'):
        '''
        Drop every index we've constructed.
        Necessary to reset the environment state.

        If the mode is 'cost' (cost estimation variant), drop the virtual indexes.
        If the mode is 'exe' (execution engine variant), drop the real indexes.
        '''
        for replica in  self.replicas:
            replica.drop_all_indexes(self.tables, mode)
    
    def _compute_baseline(self):
        benchmark_fn = None
        if self.mode == 'cost':
            benchmark_fn = self._benchmark_index_cost
        else:
            benchmark_fn = self._benchmark_index_exe

        baseline = 0
        for replica in self.replicas:
            if cost := benchmark_fn(self.queries, replica):
                baseline += cost
        
        self.baseline = baseline

    def _compute_space(self, candidates):
        return sum([self.spaces_used[x] for x in candidates])

    def _update_mask(self, replica: int):
        '''
        Update the action state mask. Marks this replica as 'complete'.
        '''
        lower_bound = replica * self.num_candidates
        upper_bound = (replica + 1) * self.num_candidates
        self._action_mask[lower_bound:upper_bound] = 0
