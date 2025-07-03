import numpy as np
import re
from util import extract_columns_from_query, construct_indexes_from_candidate, drop_one, powerset
from itertools import permutations
from profiling import Profiler
from database import Replica

class Preprocessor:
    def __init__(self, profiler: Profiler, database: Replica, max_index_width: int, queries: list[str], templates: list[int]):
        '''
        Instantiate the preprocessing module.

        `database` is the database replica we should use for querying table names,
        column names, and index sizes.
        '''
        self.columns = []
        self.tables = []
        self.workload_matrix = []
        self.access_vector = []
        self.workload = queries
        self.template_assignments = templates
        self.profiler = profiler
        self.database = database
        self.max_index_width = max_index_width

    def preprocess(self, space_budget):
        self.profiler.time_in('filesystem')
        self.workload = [self._update_query_text(query) for query in self.workload]
        self.templates = []
        for x in set(self.template_assignments):
            self.templates.append(self.workload[self.template_assignments.index(x)])
        self.profiler.time_out()
        self.profiler.time_in('database.preprocess')
        self._read_tables()
        self._read_columns()
        self.profiler.time_out()
        self.get_indexable_columns(self.templates)
        self.build_access_vector()

        print(self.candidates)

    def _read_tables(self):
        conn = self.database.connection()
        with conn.cursor() as cur:
            cur.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\';')
            for name in cur.fetchall():
                if 'hypopg' not in name[0]:
                    self.tables.append(name[0])
            conn.commit()
    
    def _read_columns(self):
        assert len(self.tables) > 0, 'trying to read columns before tables!'

        columns = set()
        self.cols_to_table = {}

        QUERY_TEMPLATE = "SELECT * FROM %s LIMIT 0;"

        try:
            conn = self.database.connection()
            with conn.cursor() as cur:
                for table in self.tables:
                    cur.execute(QUERY_TEMPLATE % table)
                    for desc in cur.description:
                        columns.add(desc[0])
                        self.cols_to_table[desc[0]] = table
                conn.commit()
        except Exception as err:
            print('got an exception in the database connection')
            print(err)
            conn.rollback()
        
        self.columns = sorted(list(columns))
        self.access_vector = [0 for _ in range(len(self.columns))]
        self.workload_matrix = np.zeros((len(self.templates), len(self.columns)))
    
    def get_indexable_columns(self, templates):
        self.candidates = {}

        for idx, template in enumerate(templates):
            matches = extract_columns_from_query(template, self.cols_to_table)
            for table, columns in matches.items():
                if table not in self.candidates:
                    self.candidates[table] = set()
                for index in powerset(sorted(columns), self.max_index_width):
                    if len(index) == 0: continue
                    for permutation in permutations(index):
                        self.candidates[table].add(permutation)
                for column in columns:
                    position = self.columns.index(column)
                    self.workload_matrix[idx][position] = 1
        
        # flatten dict of sets of tuples into a list of tuples
        self.tables = list(self.candidates.keys())
        self.candidates = list(set([x for v in self.candidates.values() for x in v]))
        self.candidates = sorted(self.candidates)
    
    def build_access_vector(self):
        for query in self.workload:
            matches = extract_columns_from_query(query, self.cols_to_table)
            for columns in matches.values():
                for column in columns:
                    position = self.columns.index(column)
                    self.access_vector[position] += 1
    
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
