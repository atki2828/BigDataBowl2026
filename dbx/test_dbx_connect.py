import os

from databricks import sql

host = os.environ["DATABRICKS_SERVER_HOSTNAME"]
http_path = os.environ["DATABRICKS_HTTP_PATH"]
access_token = os.environ["DATABRICKS_TOKEN"]

query = """SELECT *
FROM workspace.bigdatabowl2024.games
LIMIT 4"""


with sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=access_token,
) as connection:

    with connection.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
        for row in result:
            print(row)
