import sqlite3
import pandas as pd

def create_tables(conn):
    """Crée la structure des tables si elles n'existent pas déjà."""
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS passengers (
            id          INTEGER PRIMARY KEY,
            first_name  TEXT NOT NULL,
            last_name   TEXT NOT NULL,
            email       TEXT UNIQUE,
            nationality TEXT
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flights (
            id              INTEGER PRIMARY KEY,
            flight_number   TEXT NOT NULL,
            origin          TEXT NOT NULL,
            destination     TEXT NOT NULL,
            departure_time  TEXT NOT NULL,
            arrival_time    TEXT NOT NULL,
            aircraft        TEXT,
            capacity        INTEGER,
            price_eur       REAL
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id              INTEGER PRIMARY KEY,
            passenger_id    INTEGER NOT NULL,
            flight_id       INTEGER NOT NULL,
            booking_date    TEXT,
            seat_class      TEXT CHECK(seat_class IN ('economy', 'business', 'first')),
            seat_number     TEXT,
            status          TEXT CHECK(status IN ('confirmed', 'cancelled', 'pending')),
            FOREIGN KEY (passenger_id) REFERENCES passengers(id),
            FOREIGN KEY (flight_id) REFERENCES flights(id)
        );
    ''')
    conn.commit()

def insert_data(conn, data, table_name):
    """Insère des données sans lever d'erreur en cas de doublon sur la clé primaire."""
    if not data:
        return
    cursor = conn.cursor()
    placeholders = ', '.join(['?'] * len(data[0]))
    query_sql = f'INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})'
    cursor.executemany(query_sql, data)
    conn.commit()

def seed_database(conn):
    """Alimente la base de données avec le jeu de test initial."""
    passengers_data = [
        (1, 'Lucas',   'Moreau',   'lucas.moreau@email.fr',    'Française'),
        (2, 'Emma',    'Dubois',   'emma.dubois@email.fr',     'Française'),
        (3, 'Noah',    'Leroy',    'noah.leroy@email.fr',      'Française'),
        (4, 'Sofia',   'Rossi',    'sofia.rossi@email.it',     'Italienne'),
        (5, 'Liam',    'Smith',    'liam.smith@email.uk',      'Britannique'),
        (6, 'Mia',     'Müller',   'mia.muller@email.de',      'Allemande'),
        (7, 'Hugo',    'Bernard',  'hugo.bernard@email.fr',    'Française'),
        (8, 'Camille', 'Thomas',   'camille.thomas@email.fr',  'Française'),
        (9, 'Alice',   'Martin',   'alice.martin@email.fr',    'Française'),
        (10,'Carlos',  'Garcia',   'carlos.garcia@email.es',   'Espagnole'),
        (11,'Jade',    'Petit',    'jade.petit@email.fr',      'Française'),
        (12,'Omar',    'Hassan',   'omar.hassan@email.ma',     'Marocaine'),
        (13,'Elena',   'Popescu',  'elena.popescu@email.ro',   'Roumaine'),
        (14,'Antoine', 'Garnier',  'antoine.garnier@email.fr', 'Française'),
        (15,'Yako',    'Tanaka',   'yuki.tanaka@email.jp',     'Japonaise')
    ]

    flights_data = [
        (1,  'AF1234', 'Nice',      'Paris CDG',    '2026-07-01 07:00', '2026-07-01 08:30', 'Airbus A320', 180, 89.0),
        (2,  'AF1235', 'Paris CDG', 'Nice',          '2026-07-01 19:00', '2026-07-01 20:30', 'Airbus A320', 180, 95.0),
        (3,  'EZ4501', 'Nice',      'Londres',       '2026-07-02 06:30', '2026-07-02 08:15', 'Airbus A319', 156, 120.0),
        (4,  'EZ4502', 'Nice',      'Amsterdam',     '2026-07-02 09:00', '2026-07-02 11:30', 'Airbus A319', 156, 135.0),
        (5,  'VY6010', 'Nice',      'Barcelone',     '2026-07-03 11:00', '2026-07-03 12:45', 'Boeing 737',  189, 78.0),
        (6,  'AF5501', 'Nice',      'New York JFK',  '2026-07-04 13:00', '2026-07-04 22:30', 'Boeing 777',  350, 650.0),
        (7,  'LH2201', 'Nice',      'Francfort',     '2026-07-05 08:00', '2026-07-05 09:45', 'Airbus A320', 180, 112.0),
        (8,  'IB3301', 'Nice',      'Madrid',        '2026-07-05 14:00', '2026-07-05 16:00', 'Boeing 737',  189, 99.0),
        (9,  'AF1236', 'Nice',      'Paris CDG',     '2026-07-06 18:00', '2026-07-06 19:30', 'Airbus A320', 180, 105.0),
        (10, 'U24401', 'Nice',      'Athènes',       '2026-07-07 10:00', '2026-07-07 13:30', 'Boeing 737',  189, 145.0),
        (11, 'FR7701', 'Nice',      'Dublin',        '2026-07-08 07:30', '2026-07-08 10:00', 'Boeing 737',  189, 89.0),
        (12, 'AF1237', 'Paris CDG', 'Nice',          '2026-07-08 20:00', '2026-07-08 21:30', 'Airbus A321', 220, 88.0),
        (13, 'EZ4503', 'Nice',      'Berlin',        '2026-07-09 12:00', '2026-07-09 14:15', 'Airbus A319', 156, 118.0),
        (14, 'AF5502', 'Nice',      'New York JFK',  '2026-07-10 13:00', '2026-07-10 22:30', 'Boeing 777',  350, 680.0),
        (15, 'TK8801', 'Nice',      'Istanbul',      '2026-07-11 09:00', '2026-07-11 12:30', 'Boeing 737',  189, 160.0)
    ]

    bookings_data = [
        (1,  1,  1,  '2026-06-01', 'economy',  '14A', 'confirmed'),
        (2,  2,  1,  '2026-06-02', 'business', '2C',  'confirmed'),
        (4,  4,  5,  '2026-06-05', 'economy',  '18C', 'confirmed'),
        (5,  5,  3,  '2026-06-06', 'business', '1A',  'confirmed'),
        (6,  6,  7,  '2026-06-07', 'economy',  '31D', 'confirmed'),
        (7,  7,  6,  '2026-06-08', 'first',    '1A',  'confirmed'),
        (8,  8,  6,  '2026-06-09', 'business', '3B',  'confirmed'),
        (9,  9,  9,  '2026-06-10', 'economy',  '25A', 'confirmed'),
        (10, 10, 5,  '2026-06-11', 'economy',  '17B', 'cancelled'),
        (11, 11, 10, '2026-06-12', 'economy',  '8C',  'confirmed'),
        (12, 12, 15, '2026-06-13', 'economy',  '11A', 'confirmed'),
        (13, 13, 4,  '2026-06-14', 'economy',  '29B', 'pending'),
        (14, 14, 2,  '2026-06-15', 'business', '4A',  'confirmed'),
        (15, 15, 6,  '2026-06-16', 'first',    '2A',  'confirmed'),
        (16, 1,  9,  '2026-06-17', 'economy',  '33C', 'confirmed'),
        (17, 2,  10, '2026-06-18', 'economy',  '15A', 'cancelled'),
        (18, 5,  6,  '2026-06-19', 'business', '5B',  'confirmed'),
        (19, 7,  14, '2026-06-20', 'first',    '1B',  'confirmed'),
        (20, 9,  11, '2026-06-21', 'economy',  '22D', 'confirmed')
    ]

    insert_data(conn, passengers_data, 'passengers')
    insert_data(conn, flights_data, 'flights')
    insert_data(conn, bookings_data, 'bookings')

def get_connection(db_name=":memory:"):
    """Crée une connexion, initialise les tables et charge les données."""
    conn = sqlite3.connect(db_name, check_same_thread=False)
    create_tables(conn)
    seed_database(conn)
    return conn

def execute_query(conn, sql_query: str) -> pd.DataFrame:
    """Exécute une requête SQL et retourne les résultats sous forme de DataFrame Pandas."""
    return pd.read_sql_query(sql_query, conn)

