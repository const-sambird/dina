import glob
import os
import random

def load_training_set_queries(path: str, fraction: float) -> tuple[list[str], list[int]]:
    '''
    Load `fraction` of the queries in the training set located at `path`.
    Returns the query text in a list as well as which template each query belongs to.
    
    :param path: the location of the training set
    :param fraction: what proportion of the total training set should we load
    :returns queries: the query text
    :returns templates: which template each query belongs to
    '''
    all_queries = glob.glob(f'{path}/*.sql')
    n_queries = len(all_queries)
    n_selected = round(n_queries * fraction)
    query_names = [os.path.basename(q) for q in all_queries]
    selections = random.sample(query_names, n_selected)

    queries = []
    templates = []

    for selection in selections:
        template = selection.split('_')[0]
        template = int(template)
        with open(f'{path}/{selection}', 'r') as infile:
            lines = infile.readlines()
            flattened = ' '.join(lines[1:])
            queries.append(flattened.replace('\n', ' ').replace('\t', ''))
            templates.append(template)

    return queries, templates
