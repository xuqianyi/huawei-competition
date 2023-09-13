import pymysql

host = "127.0.0.1"
port = 3306
db_name = "HUAWEI"
user_name = "root"
password = "26122000"

def test():
    conn = create_connection()
    with conn.cursor() as cursor:
        sql = """
                SELECT VERSION();
              """
        cursor.execute(sql)
        data = cursor.fetchall()
        print(data)
    conn.close()

def create_connection():
    conn = pymysql.connect(host=host,
                           port=port,
                           db=db_name,
                           user=user_name,
                           password=password
                           )
    return conn


def randomly_select_data(num_question):
    conn = create_connection()
    with conn.cursor() as cursor:
        sql = "SELECT question_id, correct_answer FROM Questions ORDER BY RAND() LIMIT %s;"
        cursor.execute(sql, num_question)
        data = cursor.fetchall()
    conn.close()
    return data

def get_days(user_id):
    conn = create_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT days FROM Plan WHERE user_id=%s;"
            cursor.execute(sql, user_id)
            day = cursor.fetchall()
    except pymysql.err.OperationalError:
        conn.close()
        return -1, "Failed in inserting plan"
    conn.close()
    return day

def insert_plan(user_id, num_qn, start_date, end_date, days):
    conn = create_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO Plan (num_qn, start_date, end_date, days) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, [str(num_qn), str(start_date), str(end_date), str(days)])
            conn.commit()
    except pymysql.err.OperationalError:
        conn.close()
        return -1, "Failed in inserting plan"
    conn.close()
    return 0, "Succeed in inserting plan"

def insert_mistakes(user_id, question_ids):
    conn = create_connection()
    for question_id in question_ids:
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO Mistakes (question_id, user_id) VALUES (%s, %s)"
                cursor.execute(sql, [str(question_id), str(user_id)])
                conn.commit()
        except pymysql.err.OperationalError:
            conn.close()
            return -1, "Failed in inserting mistakes"
    conn.close()
    return 0, "Succeed in inserting mistakes"


def insert_favorite(user_id, question_ids):
    conn = create_connection()
    for question_id in question_ids:
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO Favorites (question_id, user_id) VALUES (%s, %s)"
                cursor.execute(sql, [str(question_id), str(user_id)])
                conn.commit()
        except pymysql.err.OperationalError:
            conn.close()
            return -1, "Failed in inserting favorites"
    conn.close()
    return 0, "Succeed in inserting favorites"

def select_mistakes(user_id):
    conn = create_connection()
    with conn.cursor() as cursor:
        sql =   """
                SELECT
                    question_id, correct_answer
                FROM
                    Questions
                WHERE
                    question_id in (
                        SELECT DISTINCT question_id FROM Mistakes WHERE user_id = %s)
                """
        cursor.execute(sql, [str(user_id)])
        questions = cursor.fetchall()
    conn.close()
    return questions

def select_favorites(user_id):
    conn = create_connection()
    with conn.cursor() as cursor:
        sql = """
                SELECT
                    question_id, correct_answer
                FROM
                    Questions
                WHERE
                    question_id in (
                        SELECT DISTINCT question_id FROM Favorites WHERE user_id = %s)
                """
        cursor.execute(sql, [str(user_id)])
        data = cursor.fetchall()
    conn.close()
    return data
