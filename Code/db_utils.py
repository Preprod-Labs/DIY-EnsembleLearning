# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Vansh R
        # Role: Architect
        # Code ownership rights: Vansh R
    # Version:
        # Version: V 1.0 (29 July 2024)
            # Developer: Vansh R
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script contains functions to read a YAML configuration file, connect to PostgreSQL and Cassandra databases, create tables in both databases, and insert data into these tables. The functions utilize SQLAlchemy for PostgreSQL connections and the Cassandra driver for Cassandra connections.
        # PostgreSQL: Yes
        # Cassandra: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # SQLAlchemy: 2.0.31
        # Cassandra-driver: 3.29.1

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, Date
from cassandra.cluster import Cluster

def connect_postgresql(username, password, host, port, database):
    # Function to connect to PostgreSQL database using the provided configuration
    
    engine = create_engine(f"postgresql://{username}:{password}@{host}:{port}/{database}")
    return engine  # Return SQLAlchemy engine for PostgreSQL connection

def connect_cassandra(host, port, keyspace):
    # Function to connect to Cassandra database using the provided configuration
    
    cluster = Cluster([host], port, auth_provider=None, protocol_version=4)
    session = cluster.connect(keyspace)  # Connect to the specified keyspace
    return session  # Return Cassandra session

def create_postgresql_table(engine):
    # Function to create a PostgreSQL table for customer data
    metadata = MetaData()  # Metadata object to hold information about the table
    customer_data = Table('customer_data', metadata,
                          Column('customer_id', String, primary_key=True),  # Primary key column
                          Column('subscription_type', String),  # Subscription type column
                          Column('annual_fee', Float),  # Annual fee column
                          Column('payment_method', String),  # Payment method column
                          Column('account_age', Integer),  # Account age column
                          Column('number_of_logins', Integer),  # Number of logins column
                          Column('total_spent', Float),  # Total amount spent column
                          Column('num_tickets_raised', Integer),  # Number of tickets raised column
                          Column('avg_response_time', Float),  # Average response time column
                          Column('satisfaction_score', Integer),  # Satisfaction score column
                          Column('country', String),  # Country column
                          Column('device', String),  # Device type column
                          Column('churn', Boolean))  # Churn status column
    metadata.create_all(engine)  # Create the table in the database

def create_cassandra_table(session):
    # Function to create a Cassandra table for customer data
    session.execute("""
        CREATE TABLE IF NOT EXISTS customer_data (
            customer_id TEXT PRIMARY KEY,
            signup_date DATE,
            last_login DATE,
            usage_hours_per_month INT
        )
    """)

def insert_data_to_postgresql(data, table_name, engine):
    # Function to insert data into a PostgreSQL table
    
    # Create the table if not exists
    create_postgresql_table(engine)
    
    data.to_sql(table_name, engine, if_exists='replace', index=False)  # Insert data into the specified table

def insert_data_to_cassandra(data, session, table_name):
    # Function to insert data into a Cassandra table
    
    # Create the table if not exists
    create_cassandra_table(session)
    
    for index, row in data.iterrows():  # Iterate over each row in the DataFrame
        session.execute(f"""
            INSERT INTO {table_name} (customer_id, signup_date, last_login, usage_hours_per_month)
            VALUES (%s, %s, %s, %s)
        """, (row['customer_id'], row['signup_date'], row['last_login'], row['usage_hours_per_month']))  # Insert each row into the table