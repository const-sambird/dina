import glob
import os
import subprocess
import shutil
import logging
import random
import psycopg

TABLE_BLOCK_SIZE = 5_120_000

class Replica:
    def __init__(self, id: int, hostname: str, port: str, dbname: str, user: str, password: str = ''):
        self.id = id
        self.hostname = hostname
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.connection_string = self._connection_string()

    def _connection_string(self) -> str:
        return f'host={self.hostname} port={self.port} dbname={self.dbname} user={self.user} password={self.password}'

class Connection:
    def __init__(self, replica: Replica):
        '''
        Represents a single connection to each database.

        Due to the nature of Python's multiprocessing and psycopg's
        `Connection` object, each database within each query stream will
        require its own connection to the database (at scale factor 10
        with 3 query streams and 2 replicas, that is 6 connections total,
        but they're not all going to the same database, so that isn't too
        much).

        Connections should only be created from *within* each spawned process
        (typically from the `QueryStream` class), presuming they're being used
        in a multiprocess context.

        :param replica: the database `Replica` we are connecting to
        '''
        self.replica = replica
        self._connection = psycopg.connect(replica.connection_string, autocommit=True)
    
    def conn(self):
        if self._connection is not None:
            return self._connection
        
        logging.error(f'connecting to replica {self.replica.id}: connection has already been closed!')
        return None
    
    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

class TPCHGenerator:
    def __init__(self, replicas: list[Replica], data_path: str, scale_factor: int):
        '''
        TPC-H data generator. Interface to the `dbgen` tool provided by
        the Transaction Processing Performance Council.

        `dbgen` should be downloaded in the directory given by `dbgen_path` and
        a Makefile created, but this program will recompile and execute dbgen
        with the given `scale_factor`.

        After the data is created, it can be loaded into the specified database
        replicas.

        Used QRLIT as a reference: https://github.com/DBarbosaDev/QRLIT/blob/main/db_env/tpch/TpchGenerator.py

        :param replicas: the databases
        :param dbgen_path: the location of the directory where dbgen will be compiled (fully qualified)
        :param data_path: the desired data directory for the output table data/queries/refresh functions
        :param scale_factor: the TPC-H scale factor (usually 10 for our experiments)
        '''
        self.replicas = replicas
        self.dbgen_path = None
        self.data_path = data_path
        self.scale_factor = str(scale_factor)
        self.dbname = replicas[0].dbname
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
    
    def generate(self, rng_seed: int | None = None):
        '''
        Generates the table, query, and refresh data for the
        TPC-H benchmark, according to the parameters that were
        passed to the constructor of the Generator class.

        Creates the specified data directory (if it does not already exist),
        compiles dbgen (or recompiles it), and generates the data.
        '''
        if rng_seed == None:
            rng_seed = random.randrange(1_000_000_000, 9_999_999_999)
        rng_seed = str(rng_seed)

        self._create_directories()
        self._move_query_templates()
        self._compile_dbgen()
        self._create_table_data()
        self._format_table_data()
        self._create_queries()

    def load_database(self):
        '''
        Loads the data into the database.

        (1) Resets each database, creates the schemas, and loads the generated table data.

        (2) Creates the default primary key/foreign key constraints on those tables.
        '''
        connections = [Connection(replica) for replica in self.replicas]
        tables = []

        for table_file in glob.glob(f'{self.data_path}/tables/*.tbl'):
            name = os.path.basename(table_file)
            tables.append(name.split('.')[0])

        self._reset_database(connections, tables)
        self._create_schemas(connections)
        self._load_table_data(connections)
        self._create_keys(connections)

        for c in connections:
            c.close()

    def read_data(self) -> tuple[list[str], list[int]]:
        '''
        Loads the queries into memory.

        :returns queries: the 22 generated queries to be executed according to the stream order given in the specification
        :returns templates: the template number of each query
        '''
        return self._load_queries()
    
    def _create_directories(self):
        logging.debug(f'creating data directories under {self.data_path}')
        os.makedirs(f'{self.data_path}/refresh', exist_ok=True)
        os.makedirs(f'{self.data_path}/tables', exist_ok=True)
        os.makedirs(f'{self.data_path}/queries', exist_ok=True)
        os.makedirs(f'{self.data_path}/schema', exist_ok=True)

    def _compile_dbgen(self):
        logging.debug(f'attempting to compile TPC-H dbgen at {self.dbgen_path}')
        subprocess.run('make', cwd=self.dbgen_path)
    
    def _create_table_data(self):
        logging.debug(f'creating table data for scale factor {self.scale_factor}')
        subprocess.run([f'{self.dbgen_path}/dbgen', '-s', self.scale_factor, '-vf'], cwd=self.dbgen_path)

        table_paths = glob.glob('*.tbl', root_dir=self.dbgen_path)

        for file in table_paths:
            shutil.move(f'{self.dbgen_path}/{file}', f'{self.data_path}/tables/{os.path.basename(file)}')

        shutil.copy(f'{self.dbgen_path}/dss.ddl', f'{self.data_path}/schema/dss.ddl')
        shutil.copy(f'{self.root_dir}/schema_keys.sql', f'{self.data_path}/schema/schema_keys.sql')
    
    def _move_query_templates(self):
        existing_templates = glob.glob(f'{self.dbgen_path}/queries/*.sql')
        corrected_templates = glob.glob(f'{self.root_dir}/tpch-templates/*.sql')

        for template in existing_templates:
            os.remove(template)
        
        for template in corrected_templates:
            shutil.copy(template, f'{self.dbgen_path}/queries')

    def _create_queries(self):
        logging.debug(f'creating TPC-H query data')

        for i in range(1, 23):
            with open(f'{self.data_path}/queries/{i}.sql', 'w') as outfile:
                subprocess.run([f'{self.dbgen_path}/qgen', '-s', self.scale_factor, str(i)],
                               cwd=self.dbgen_path,
                               env=dict(os.environ, DSS_QUERY=f'{self.dbgen_path}/queries'),
                               stdout=outfile)
    
    def _reset_database(self, connections: list[Connection], tables: list[str]):
        '''
        Drops the tables specified.

        :param tables: a list of table names
        '''
        logging.debug(f'dropping existing tables: {tables}')
        for c in connections:
            with c.conn().cursor() as cur:
                for table in tables:
                    cur.execute(f'DROP TABLE IF EXISTS {table} CASCADE')

    def _create_schemas(self, connections: list[Connection]):
        logging.info('creating the schemas for tables')
        for c in connections:
            with c.conn().cursor() as cur:
                with open(f'{self.data_path}/schema/dss.ddl', 'r') as infile:
                    cur.execute(infile.read())

    def _create_keys(self, connections: list[Connection]):
        logging.info('creating primary and foreign keys')
        for c in connections:
            with c.conn().cursor() as cur:
                with open(f'{self.data_path}/schema/schema_keys.sql', 'r') as infile:
                    cur.execute(infile.read())
    
    def _format_table_data(self):
        '''
        The table data by default includes a trailing pipe (|) character
        that must be removed for Postgres to process it correctly with the
        `COPY FROM ... FORMAT CSV` command. We do that in-place here.
        '''
        logging.info('correcting table data CSV format for postgres...')
        for table_file in glob.glob(f'*.tbl', root_dir=f'{self.data_path}/tables'):
            logging.debug(table_file)
            subprocess.run(['sed', '-i', 's/.$//', table_file], cwd=f'{self.data_path}/tables')
    
    def _load_table_data(self, connections: list[Connection]):
        for table_file in glob.glob(f'{self.data_path}/tables/*.tbl'):
            table = os.path.basename(table_file).split('.')[0]
            logging.info(f'loading data into {table}')
            for num, c in enumerate(connections):
                logging.debug(f'loading to replica {num}')
                with c.conn().cursor() as cur:
                    with cur.copy(f'COPY {table} FROM STDIN (format csv, delimiter \'|\')') as copy:
                        with open(table_file, 'r') as input:
                            while data := input.read(TABLE_BLOCK_SIZE):
                                copy.write(data)

    def _load_queries(self) -> tuple[list[str], list[int]]:
        logging.info('reading queries')
        queries = []
        templates = list(range(22))

        for i in range(1, 23):
            with open(f'{self.data_path}/queries/{i}.sql', 'r') as infile:
                queries.append(infile.read())
        
        return queries, templates
