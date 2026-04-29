from sqlalchemy import create_engine, inspect, text

from app.db.schema import ensure_runtime_schema


def test_ensure_runtime_schema_adds_chat_message_payload_columns_and_drops_legacy_tables():
    engine = create_engine("sqlite+pysqlite:///:memory:")

    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id VARCHAR PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE roles (id VARCHAR PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE user_roles (id VARCHAR PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE user_client_access (id VARCHAR PRIMARY KEY)"))

        conn.execute(
            text(
                """
                CREATE TABLE clients (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    description VARCHAR,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE chat_sessions (
                    id VARCHAR PRIMARY KEY,
                    client_id VARCHAR NOT NULL,
                    title VARCHAR,
                    summary_text VARCHAR,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    last_activity_at DATETIME NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE query_logs (
                    id VARCHAR PRIMARY KEY,
                    client_id VARCHAR NOT NULL,
                    question VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    created_at DATETIME NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE chat_messages (
                    id VARCHAR PRIMARY KEY,
                    client_id VARCHAR NOT NULL,
                    session_id VARCHAR NOT NULL,
                    role VARCHAR NOT NULL,
                    content VARCHAR NOT NULL,
                    turn_index INTEGER NOT NULL,
                    query_log_id VARCHAR,
                    created_at DATETIME NOT NULL
                )
                """
            )
        )

    ensure_runtime_schema(engine)
    ensure_runtime_schema(engine)

    schema = inspect(engine)
    chat_columns = {column["name"] for column in schema.get_columns("chat_messages")}
    table_names = set(schema.get_table_names())

    assert "reasoning" in chat_columns
    assert "citations_json" in chat_columns
    assert "users" not in table_names
    assert "roles" not in table_names
    assert "user_roles" not in table_names
    assert "user_client_access" not in table_names
