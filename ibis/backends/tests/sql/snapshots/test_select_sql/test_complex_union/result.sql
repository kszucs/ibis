SELECT
  CAST(t0.diag + CAST(1 AS TINYINT) AS INT) AS diag,
  t0.status AS status
FROM aids2_one AS t0
UNION ALL
SELECT
  CAST(t1.diag + CAST(1 AS TINYINT) AS INT) AS diag,
  t1.status AS status
FROM aids2_two AS t1