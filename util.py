from itertools import chain, combinations
import re

def powerset(iterable):
    # https://docs.python.org/2/library/itertools.html#recipes
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def extract_table_from_query(query: str) -> str | None:
        REGEX = '^(([a-zA-Z0-9]|\\.|_|-)+) '
        split = query.split('FROM ')
        
        if len(split) != 2:
            return None # subquery or malformed
        if table_name := re.search(REGEX, split[1], re.IGNORECASE):
            return table_name.group(1)
        else:
            return None
