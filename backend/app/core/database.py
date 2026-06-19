"""
database.py
============
SQLAlchemy data layer. Defaults to a local SQLite file so the project
runs out-of-the-box with zero external services. Set the environment
variable DATABASE_URL to point at MySQL instead, e.g.:

    export DATABASE_URL="mysql+pymysql://user:password@localhost:3306/xnids"

and `pip install pymysql` (already listed in requirements.txt).

Nothing else needs to be created by hand: `init_db()` (called once at
FastAPI startup, see main.py) first makes sure the target *database*
itself exists (`ensure_database_exists()`), then creates every table via
`Base.metadata.create_all()`. For SQLite the database is just a file
that's created automatically on first connection, so this is a no-op
there. For MySQL we open a throwaway connection to the server (no
database selected) and issue `CREATE DATABASE IF NOT EXISTS`, so a fresh
MySQL server with just a user/password and no pre-existing schema works
out of the box too.
"""

import os
from datetime import datetime

from sqlalchemy import (create_engine, text, Column, Integer, String, Float,
                         DateTime, Text, Boolean)
from sqlalchemy.engine import make_url, URL
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./xnids.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AttackLog(Base):
    __tablename__ = "attack_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    src_ip = Column(String(64))
    dst_ip = Column(String(64))
    src_mac = Column(String(64))
    dst_port = Column(Integer)
    protocol = Column(String(16))
    attack_type = Column(String(64))
    severity = Column(String(16))
    status = Column(String(32), default="active")  # active | mitigated
    model_used = Column(String(64))
    confidence = Column(Float)
    anomaly_score = Column(Float)
    top_features_json = Column(Text)


class DefenseRuleRecord(Base):
    __tablename__ = "defense_rule"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    attack_log_id = Column(Integer)
    scope = Column(String(32))
    strategy = Column(String(32))
    openflow_rule = Column(Text)
    iptables_rule = Column(Text)
    pfsense_rule = Column(Text)
    active = Column(Boolean, default=True)


class WhitelistEntry(Base):
    __tablename__ = "whitelist"

    id = Column(Integer, primary_key=True, index=True)
    ip = Column(String(64), unique=True)
    label = Column(String(128), default="")
    created_at = Column(DateTime, default=datetime.utcnow)


def ensure_database_exists():
    """Creates the target database itself if it doesn't exist yet.

    SQLite: no-op (the file is created automatically on first connection).
    MySQL: connects to the server with no database selected and runs
    `CREATE DATABASE IF NOT EXISTS <name> CHARACTER SET utf8mb4`, using a
    short-lived engine that's disposed immediately after.
    """
    url = make_url(DATABASE_URL)
    backend = url.get_backend_name()

    if backend == "sqlite":
        return  # nothing to do — the file-based DB is created on connect

    if backend == "mysql":
        db_name = url.database
        server_url = URL.create(
            drivername=url.drivername,
            username=url.username,
            password=url.password,
            host=url.host,
            port=url.port,
            query=url.query,
        )
        tmp_engine = create_engine(server_url)
        try:
            with tmp_engine.connect() as conn:
                conn.execute(text(
                    f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                    f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                ))
                conn.commit()
        finally:
            tmp_engine.dispose()
        return

    # Other backends (e.g. Postgres) aren't wired up here — fall through
    # and let create_all() surface a clear connection error if the
    # database doesn't already exist.


def init_db():
    ensure_database_exists()
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

