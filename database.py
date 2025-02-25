class Replica:
    def __init__(self, id, hostname, port = 5432, dbname = 'tpchdb', user = 'sam'):
        self.id = id
        self.hostname = hostname
        self.port = port
        self.dbname = dbname
        self.user = user

    def connection_string(self):
        return f'host={self.hostname} port={self.port} dbname={self.dbname} user={self.user}'
