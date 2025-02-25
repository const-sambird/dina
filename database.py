def connection_string(hostname, port=5432, dbname='tpchdb', user='sam'):
    return f'host={hostname} port={port} dbname={dbname} user={user}'
