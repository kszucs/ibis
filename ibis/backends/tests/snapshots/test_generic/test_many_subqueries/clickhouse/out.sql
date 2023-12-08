SELECT
  t4.street AS street,
  t4.key AS key,
  t4.key_right AS key_right
FROM (
  SELECT
    t1.street AS street,
    ROW_NUMBER() OVER (ORDER BY t1.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key,
    t2.key AS key_right
  FROM (
    SELECT
      t0.street AS street,
      ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key
    FROM data AS t0
  ) AS t1
  INNER JOIN (
    SELECT
      t1.key AS key
    FROM (
      SELECT
        t0.street AS street,
        ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key
      FROM data AS t0
    ) AS t1
  ) AS t2
    ON t1.key = t2.key
) AS t4
INNER JOIN (
  SELECT
    t4.key AS key
  FROM (
    SELECT
      t1.street AS street,
      ROW_NUMBER() OVER (ORDER BY t1.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key,
      t2.key AS key_right
    FROM (
      SELECT
        t0.street AS street,
        ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key
      FROM data AS t0
    ) AS t1
    INNER JOIN (
      SELECT
        t1.key AS key
      FROM (
        SELECT
          t0.street AS street,
          ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key
        FROM data AS t0
      ) AS t1
    ) AS t2
      ON t1.key = t2.key
  ) AS t4
) AS t5
  ON t4.key = t5.key