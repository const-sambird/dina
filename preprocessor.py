import pickle
import psycopg
import re
from util import extract_columns_from_query, construct_indexes_from_candidate, drop_one, powerset
from profiling import Profiler
from database import Replica

QUERY_TEMPLATE_PATH     = './QueryBot5000/templates.txt'
CLUSTER_ASSIGNMENT_PATH = './QueryBot5000/online-clustering-results/None-0.8-assignments.pickle'
COVERAGE_PATH           = './QueryBot5000/cluster-coverage/coverage.pickle'

class Preprocessor:
    def __init__(self, profiler: Profiler, database: Replica, max_index_width: int):
        '''
        Instantiate the preprocessing module.

        `database` is the database replica we should use for querying table names,
        column names, and index sizes.
        '''
        self.columns = []
        self.workload = []
        self.profiler = profiler
        self.database = database
        self.max_index_width = max_index_width
    
    def _load_clusters(self):
        try:
            with open(CLUSTER_ASSIGNMENT_PATH, 'rb') as clusterfile:
                num_clusters, assignment_dict, cluster_totals = pickle.load(clusterfile)
        except:
            pass

    def preprocess(self, space_budget):
        self.profiler.time_in('filesystem')
        self.templates = self._read_templates()
        self.templates = [self._update_query_text(template) for template in self.templates]
        self.profiler.time_out()
        self.profiler.time_in('database.preprocess')
        self._read_tables()
        self._read_columns()
        self.profiler.time_out()
        self.get_indexable_columns(self.templates)

        print(self.candidates)
    
    def _read_templates(self, path = './templates.txt'):
        with open(path, 'r') as infile:
            return infile.readlines()

    def _read_tables(self):
        conn = self.database.connection()
        with conn.cursor() as cur:
            cur.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\';')
            self.tables = [name[0] for name in cur.fetchall()]
            conn.commit()
    
    def _read_columns(self):
        assert len(self.tables) > 0, 'trying to read columns before tables!'

        self.columns = []
        self.cols_to_table = {}

        QUERY_TEMPLATE = "SELECT * FROM %s LIMIT 0;"

        try:
            conn = self.database.connection()
            with conn.cursor() as cur:
                for table in self.tables:
                    cur.execute(QUERY_TEMPLATE % table)
                    for desc in cur.description:
                        self.columns.append(desc[0])
                        self.cols_to_table[desc[0]] = table
                conn.commit()
        except Exception as err:
            print('got an exception in the database connection')
            print(err)
            conn.rollback()
    
    def get_indexable_columns(self, templates):
        self.candidates = {}

        for idx, template in enumerate(templates):
            matches = extract_columns_from_query(template, self.cols_to_table)
            for table, columns in matches.items():
                if table not in self.candidates:
                    self.candidates[table] = set()
                for index in powerset(sorted(columns), self.max_index_width):
                    if len(index) == 0: continue
                    self.candidates[table].add(index)
        
        # flatten dict of sets of tuples into a list of tuples
        self.tables = list(self.candidates.keys())
        self.candidates = list(set([x for v in self.candidates.values() for x in v]))
        self.candidates = sorted(self.candidates)
    
    # Updates query syntax to work in PostgreSQL
    def _update_query_text(self, text: str) -> str:
        '''
        Updates query text to work in PostgreSQL.

        Taken from https://github.com/hyrise/index_selection_evaluation

        :param text: the text of the query to update
        :returns text: the corrected version
        '''
        text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
        text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)
        text = self._add_alias_subquery(text)
        return text

    # PostgreSQL requires an alias for subqueries
    def _add_alias_subquery(self, query_text):
        text = query_text.lower()
        positions = []
        for match in re.finditer(r"((from)|,)[  \n]*\(", text):
            counter = 1
            pos = match.span()[1]
            while counter > 0:
                char = text[pos]
                if char == "(":
                    counter += 1
                elif char == ")":
                    counter -= 1
                pos += 1
            next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
            if next_word[0] in [")", ","] or next_word in [
                "limit",
                "group",
                "order",
                "where",
            ]:
                positions.append(pos)
        for pos in sorted(positions, reverse=True):
            query_text = query_text[:pos] + " as alias123 " + query_text[pos:]
        return query_text
