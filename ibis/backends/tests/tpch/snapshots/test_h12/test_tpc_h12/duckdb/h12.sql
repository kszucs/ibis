SELECT
  *
FROM (
  SELECT
    t2.l_shipmode,
    SUM(
      CASE t2.o_orderpriority
        WHEN '1-URGENT'
        THEN CAST(1 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(1 AS TINYINT)
        ELSE CAST(0 AS TINYINT)
      END
    ) AS high_line_count,
    SUM(
      CASE t2.o_orderpriority
        WHEN '1-URGENT'
        THEN CAST(0 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(0 AS TINYINT)
        ELSE CAST(1 AS TINYINT)
      END
    ) AS low_line_count
  FROM (
    SELECT
      t0.*,
      t1.*
    FROM "orders" AS t0
    INNER JOIN "lineitem" AS t1
      ON (
        t0.o_orderkey = t1.l_orderkey
      )
  ) AS t2
  WHERE
    t2.l_shipmode IN ('MAIL', 'SHIP')
    AND (
      t2.l_commitdate < t2.l_receiptdate
    )
    AND (
      t2.l_shipdate < t2.l_commitdate
    )
    AND (
      t2.l_receiptdate >= MAKE_DATE(1994, 1, 1)
    )
    AND (
      t2.l_receiptdate < MAKE_DATE(1995, 1, 1)
    )
  GROUP BY
    1
) AS t3
ORDER BY
  t3.l_shipmode ASC