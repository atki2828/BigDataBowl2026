from utility.dbx import DatabricksSQLClient


def try_query():  # Named Try to not trigger pytest
    client = DatabricksSQLClient()
    query = """SELECT *
        FROM workspace.bigdatabowl2024.games
        LIMIT 4"""
    result = client.query_to_pl(query)
    print(result.head())
    return result


if __name__ == "__main__":
    try_query()
