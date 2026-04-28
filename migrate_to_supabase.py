import psycopg2
from psycopg2.extras import RealDictCursor

# Connection strings
RENDER_URL = "postgresql://english_db_yehx_user:a3ta8RP4Ej982L1xNnVykZyiER3AQPnb@dpg-d7mbgsgsfn5c73df2q6g-a.oregon-postgres.render.com/english_db_yehx"
SUPABASE_URL = "postgresql://postgres.mnrlvxeenvapwkohtijp:Z4Sc7lERLNA8qt8m@aws-1-ca-central-1.pooler.supabase.com:5432/postgres"

def migrate():
    # Connect to both databases
    render_conn = psycopg2.connect(RENDER_URL)
    supabase_conn = psycopg2.connect(SUPABASE_URL)

    render_cur = render_conn.cursor(cursor_factory=RealDictCursor)
    supabase_cur = supabase_conn.cursor()

    print("Connected to both databases.\n")

    # Step 1: Create tables in Supabase
    print("Creating tables in Supabase...")

    create_tables_sql = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(150) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        role VARCHAR(20) DEFAULT 'student',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) NOT NULL,
        topic VARCHAR(100),
        session_number INTEGER,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS turns (
        id SERIAL PRIMARY KEY,
        session_id INTEGER REFERENCES sessions(id) NOT NULL,
        turn_number INTEGER,
        app_question TEXT,
        student_speech TEXT,
        fluency_comment TEXT
    );

    CREATE TABLE IF NOT EXISTS session_analysis (
        id SERIAL PRIMARY KEY,
        session_id INTEGER REFERENCES sessions(id) NOT NULL,
        vocabulary_score FLOAT,
        vocabulary_note TEXT,
        phrasing_score FLOAT,
        phrasing_note TEXT,
        structure_score FLOAT,
        structure_note TEXT,
        overall_score FLOAT,
        overall_note TEXT,
        suggestion TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS progress_reports (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) NOT NULL,
        report_number INTEGER,
        sessions_from INTEGER,
        sessions_to INTEGER,
        vocabulary_score FLOAT,
        vocabulary_label VARCHAR(20),
        vocabulary_description TEXT,
        phrasing_score FLOAT,
        phrasing_label VARCHAR(20),
        phrasing_description TEXT,
        structure_score FLOAT,
        structure_label VARCHAR(20),
        structure_description TEXT,
        overall_score FLOAT,
        overall_label VARCHAR(20),
        improvement_description TEXT,
        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    supabase_cur.execute(create_tables_sql)
    supabase_conn.commit()
    print("Tables created.\n")

    # Step 2: Migrate data table by table (in order due to foreign keys)

    # Migrate users
    print("Migrating users...")
    render_cur.execute("SELECT * FROM users ORDER BY id")
    users = render_cur.fetchall()
    for u in users:
        supabase_cur.execute("""
            INSERT INTO users (id, name, email, password_hash, role, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (u['id'], u['name'], u['email'], u['password_hash'], u['role'], u['created_at']))
    supabase_conn.commit()
    print(f"  Migrated {len(users)} users")

    # Migrate sessions
    print("Migrating sessions...")
    render_cur.execute("SELECT * FROM sessions ORDER BY id")
    sessions = render_cur.fetchall()
    for s in sessions:
        supabase_cur.execute("""
            INSERT INTO sessions (id, user_id, topic, session_number, date)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (s['id'], s['user_id'], s['topic'], s['session_number'], s['date']))
    supabase_conn.commit()
    print(f"  Migrated {len(sessions)} sessions")

    # Migrate turns
    print("Migrating turns...")
    render_cur.execute("SELECT * FROM turns ORDER BY id")
    turns = render_cur.fetchall()
    for t in turns:
        supabase_cur.execute("""
            INSERT INTO turns (id, session_id, turn_number, app_question, student_speech, fluency_comment)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (t['id'], t['session_id'], t['turn_number'], t['app_question'], t['student_speech'], t['fluency_comment']))
    supabase_conn.commit()
    print(f"  Migrated {len(turns)} turns")

    # Migrate session_analysis
    print("Migrating session_analysis...")
    render_cur.execute("SELECT * FROM session_analysis ORDER BY id")
    analyses = render_cur.fetchall()
    for a in analyses:
        supabase_cur.execute("""
            INSERT INTO session_analysis (id, session_id, vocabulary_score, vocabulary_note,
                phrasing_score, phrasing_note, structure_score, structure_note,
                overall_score, overall_note, suggestion, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (a['id'], a['session_id'], a['vocabulary_score'], a['vocabulary_note'],
              a['phrasing_score'], a['phrasing_note'], a['structure_score'], a['structure_note'],
              a['overall_score'], a['overall_note'], a['suggestion'], a['created_at']))
    supabase_conn.commit()
    print(f"  Migrated {len(analyses)} session analyses")

    # Migrate progress_reports
    print("Migrating progress_reports...")
    render_cur.execute("SELECT * FROM progress_reports ORDER BY id")
    reports = render_cur.fetchall()
    for r in reports:
        supabase_cur.execute("""
            INSERT INTO progress_reports (id, user_id, report_number, sessions_from, sessions_to,
                vocabulary_score, vocabulary_label, vocabulary_description,
                phrasing_score, phrasing_label, phrasing_description,
                structure_score, structure_label, structure_description,
                overall_score, overall_label, improvement_description, generated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (r['id'], r['user_id'], r['report_number'], r['sessions_from'], r['sessions_to'],
              r['vocabulary_score'], r['vocabulary_label'], r['vocabulary_description'],
              r['phrasing_score'], r['phrasing_label'], r['phrasing_description'],
              r['structure_score'], r['structure_label'], r['structure_description'],
              r['overall_score'], r['overall_label'], r['improvement_description'], r['generated_at']))
    supabase_conn.commit()
    print(f"  Migrated {len(reports)} progress reports")

    # Reset sequences to continue after max id
    print("\nResetting sequences...")
    for table in ['users', 'sessions', 'turns', 'session_analysis', 'progress_reports']:
        supabase_cur.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}")
        max_id = supabase_cur.fetchone()[0]
        supabase_cur.execute(f"SELECT setval('{table}_id_seq', %s, true)", (max_id,))
    supabase_conn.commit()
    print("Sequences reset.")

    # Close connections
    render_cur.close()
    render_conn.close()
    supabase_cur.close()
    supabase_conn.close()

    print("\n✓ Migration complete!")

if __name__ == "__main__":
    migrate()
