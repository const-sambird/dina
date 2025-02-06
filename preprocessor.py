import pickle
import psycopg
import re
from util import powerset, extract_table_from_query

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
    
    def _read_templates(self, path = './QueryBot5000/templates.txt'):
        with open(path, 'r') as infile:
            return infile.readlines()

    def _read_tables(self, templates):
        tables = set()
        next_templates = []

        for template in templates:
            if table_name := extract_table_from_query(template):
                if not table_name.startswith('pg_'): # ignore system tables
                    tables.add(table_name)
                    next_templates.append(template)
        
        self.tables = list(tables)
        self.templates = next_templates
    
    def _read_columns(self):
        assert len(self.tables) > 0, 'trying to read columns before tables!'

        self.columns = {}

        QUERY_TEMPLATE = "SELECT * FROM %s LIMIT 0;"

        with psycopg.connect('dbname=dina user=sam') as conn:
            with conn.cursor() as cur:
                for table in self.tables:
                    cur.execute(QUERY_TEMPLATE % table)
                    self.columns[table] = [desc[0] for desc in cur.description]
    
    def get_indexable_columns(self, templates):
        self.indexable = {}

        for template in templates:
            if table_name := extract_table_from_query(template):
                if table_name not in self.indexable:
                    self.indexable[table_name] = set()
                
                split = template.split(' WHERE ')
                if len(split) != 2:
                    continue

                for column in self.columns[table_name]:
                    if column in split[1]:
                        self.indexable[table_name].add(column)
    
    def get_candidate_indexes(self, space_budget):
        self.candidates = {}
        self.candidate_sizes = {}

        with psycopg.connect('dbname=dina user=sam') as conn:
            with conn.cursor() as cur:
                for table, cols in self.indexable.items():
                    self.candidates[table] = []
                    for candidate in powerset(cols):
                        if len(candidate) == 0: continue
                        select_table = table
                        if '.' in table:
                            select_table = table.split('.')[0]
                        cur.execute('CREATE INDEX candidate_index ON %s (%s);' % (table, ', '.join(candidate)))
                        cur.execute("SELECT pg_table_size('%s.candidate_index');" % select_table)
                        computed_size = cur.fetchone()[0]
                        if space_budget > computed_size:
                            self.candidates[table].append(candidate)
                            self.candidate_sizes[candidate] = computed_size
                        cur.execute('DROP INDEX "%s"."candidate_index";' % select_table)
