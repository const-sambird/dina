import pickle
import psycopg
import re
from util import powerset, extract_columns_from_query, construct_indexes_from_candidate

QUERY_TEMPLATE_PATH     = './QueryBot5000/templates.txt'
CLUSTER_ASSIGNMENT_PATH = './QueryBot5000/online-clustering-results/None-0.8-assignments.pickle'
COVERAGE_PATH           = './QueryBot5000/cluster-coverage/coverage.pickle'

class Preprocessor:
    def __init__(self):
        '''
        Instantiate the preprocessing module.

        connection is a 
        '''
        self.columns = []
        self.workload = []
        # self.connection = connection
    
    def _load_clusters(self):
        try:
            with open(CLUSTER_ASSIGNMENT_PATH, 'rb') as clusterfile:
                num_clusters, assignment_dict, cluster_totals = pickle.load(clusterfile)
        except:
            pass

    def build_workload_matrix(self, space_budget):
        self.templates = self._read_templates()
        self._read_tables(self.templates)
        self._read_columns()
        self.get_indexable_columns(self.templates)
        self.get_candidate_indexes(space_budget)

        print(self.candidates)
    
    def _read_templates(self, path = './QueryBot5000/templates.txt'):
        with open(path, 'r') as infile:
            return infile.readlines()

    def _read_tables(self, templates):
        with psycopg.connect('dbname=tpchdb user=sam') as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\';')
                self.tables = [name[0] for name in cur.fetchall()]
    
    def _read_columns(self):
        assert len(self.tables) > 0, 'trying to read columns before tables!'

        self.columns = []
        self.cols_to_table = {}

        QUERY_TEMPLATE = "SELECT * FROM %s LIMIT 0;"

        try:
            with psycopg.connect('dbname=tpchdb user=sam') as conn:
                with conn.cursor() as cur:
                    for table in self.tables:
                        cur.execute(QUERY_TEMPLATE % table)
                        for desc in cur.description:
                            self.columns.append(desc[0])
                            self.cols_to_table[desc[0]] = table
        except Exception as err:
            print('got an exception in the database connection')
            print(err)
    
    def get_indexable_columns(self, templates):
        self.indexable = set()

        for template in templates:
            matches = extract_columns_from_query(template, self.cols_to_table)
            for column in matches.values():
                self.indexable = self.indexable.union(column)
    
    def get_candidate_indexes(self, space_budget):
        self.candidates = []
        self.candidate_sizes = {}

        try:
            with psycopg.connect('dbname=tpchdb user=sam') as conn:
                with conn.cursor() as cur:
                    for candidate in self.indexable:
                        candidate = (candidate,) # FIXME
                        if len(candidate) == 0: continue
                        print('evaluating candidate:', candidate)
                        computed_size = 0
                        candidate_representation = construct_indexes_from_candidate(candidate, self.cols_to_table)
                        print('---', candidate_representation)
                        for table, columns in candidate_representation.items():
                            cur.execute('CREATE INDEX candidate_index ON %s (%s);' % (table, ', '.join(columns)))
                            cur.execute("SELECT pg_table_size('candidate_index');")
                            computed_size += cur.fetchone()[0]
                            cur.execute('DROP INDEX candidate_index;')
                        print('--- computed size:', computed_size)
                        if space_budget > computed_size:
                            self.candidates.append(candidate)
                            self.candidate_sizes[candidate] = computed_size
        except Exception as err:
            print('got an exception in the database connection')
            print(err)
