from itertools import chain, combinations
from random import randrange
import re

def powerset(iterable):
    # https://docs.python.org/2/library/itertools.html#recipes
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def extract_table_from_query(query: str) -> str | None:
        REGEX = '^( )*(([a-zA-Z0-9]|\\.|_|-)+)( )*'
        split = re.split('FROM ', query, flags=re.IGNORECASE)
        
        #if len(split) != 2:
        #    return None # subquery or malformed
        if table_name := re.search(REGEX, split[1], re.IGNORECASE):
            return table_name.group(1)
        else:
            return None
        
def extract_table_from_query_aa(query, coldict):
    for t, cols in coldict.items():
        if (query,) in cols:
            return t
        
def extract_columns_from_query(query, coldict):
    REGEX = 'WHERE (.+?) (?:\\)|group by|order by)'
    columns = {}

    predicates = re.findall(REGEX, query, re.IGNORECASE)

    for predicate in predicates:
        for column, table in coldict.items():
            if column in predicate:
                if table not in columns:
                    columns[table] = set()
                columns[table].add(column)
    
    return columns

def construct_indexes_from_candidate(candidate: tuple[str], cols_to_table: dict[str, str]) -> dict[str, list[str]]:
    '''
    Given a list of columns to index over, returns a dict
    of those columns sorted into the tables they belong to.
    '''
    candidate_representation = {}
    for column in candidate:
        table = cols_to_table[column]
        if table not in candidate_representation:
            candidate_representation[table] = []
        candidate_representation[table].append(column)
    return candidate_representation

def insert_dummy_values(template):
    HASH_VALUES = '\\$?\\@\\@\\@'
    STRING_VALUES = '\\$?\\&\\&\\&'
    INT_VALUES = '\\$?\\^\\^\\^'

    template = re.sub(HASH_VALUES, "'hash'", template)
    template = re.sub(STRING_VALUES, "'foo'", template)
    template = re.sub(INT_VALUES, '-1', template)

    return template

def drop_one(from_tuple):
    els = list(from_tuple)
    to_drop = randrange(0, len(els))
    del els[to_drop]
    return tuple(els)
