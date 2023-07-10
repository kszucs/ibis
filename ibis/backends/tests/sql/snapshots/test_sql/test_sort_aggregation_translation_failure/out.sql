SELECT
  *
FROM (
  SELECT
    t0.string_col AS string_col,
    MAX(t0.double_col) AS foo
  FROM functional_alltypes AS t0
  GROUP BY
    1
) AS t1
ORDER BY
  t1.foo DESC