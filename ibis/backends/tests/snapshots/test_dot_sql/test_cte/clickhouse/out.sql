WITH "foo" AS (
  SELECT
    *
  FROM "test_clickhouse_temp_mem_t_for_cte" AS "t0"
)
SELECT
  COUNT(*) AS "x"
FROM "foo"