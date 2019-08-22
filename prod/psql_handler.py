#!/usr/bin/python
# -*- coding: utf-8 -*-

import psycopg2
from sqlalchemy import create_engine
import sys
import pandas as pd

con = None

"""
DATABASE = {
    'drivername': 'postgres',
    'host': 'localhost',
    'port': '5432', # or 5433 for results
    'username': 'pguser',
    'password': 'pguser',
    'database': 'pgdb',
}
"""
def connect_to_db(port, db):
    settings = 'postgresql+psycopg2://pguser:pguser@localhost:{port}/{db}'.format(port=port, db=db)
    engine = create_engine(settings)
    # con = psycopg2.connect("host='localhost' dbname='pgdb' user='pguser' password='pguser'")
    # if cursor:
    # cur = con.cursor()
        # return cur
    return engine


def read_data(query, port, db):
    try:
        engine = connect_to_db(port, db)
        df = pd.read_sql_query(query, engine)
        # print(df.head())

    except (psycopg2.DatabaseError):
        if engine:
            engine.dispose()
        
        print('Error')
        sys.exit(1)
        
    finally:   
        if engine:
            engine.dispose()
        return df

def write_data(df, table, ind_col, port, db):
    try:
        engine = connect_to_db(port, db)

        df = df.set_index(ind_col)
        df.to_sql(name=table, con=engine, if_exists='append')

        # con.commit()
    except (psycopg2.DatabaseError):
        if engine:
            engine.dispose()
        
        print('Error')
        sys.exit(1)
        
    finally:   
        if engine:
            engine.dispose()
