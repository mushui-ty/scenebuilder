
import lancedb


db_uri = "/data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/manycore"

table_name = "furniture"



db = lancedb.connect(db_uri)
table = db.open_table(table_name)


query_line = table.to_pandas().iloc[0]
query_vector = query_line['vector']
print(query_line)

result = table.search(query_vector).select(["asset_id", "category_zh", "label"]).limit(10).to_polars()
print(result)