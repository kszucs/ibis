SELECT
  "t25"."n_name",
  "t25"."revenue"
FROM (
  SELECT
    "t24"."n_name",
    SUM("t24"."l_extendedprice" * (
      1 - "t24"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t23"."c_custkey",
      "t23"."c_name",
      "t23"."c_address",
      "t23"."c_nationkey",
      "t23"."c_phone",
      "t23"."c_acctbal",
      "t23"."c_mktsegment",
      "t23"."c_comment",
      "t23"."o_orderkey",
      "t23"."o_custkey",
      "t23"."o_orderstatus",
      "t23"."o_totalprice",
      "t23"."o_orderdate",
      "t23"."o_orderpriority",
      "t23"."o_clerk",
      "t23"."o_shippriority",
      "t23"."o_comment",
      "t23"."l_orderkey",
      "t23"."l_partkey",
      "t23"."l_suppkey",
      "t23"."l_linenumber",
      "t23"."l_quantity",
      "t23"."l_extendedprice",
      "t23"."l_discount",
      "t23"."l_tax",
      "t23"."l_returnflag",
      "t23"."l_linestatus",
      "t23"."l_shipdate",
      "t23"."l_commitdate",
      "t23"."l_receiptdate",
      "t23"."l_shipinstruct",
      "t23"."l_shipmode",
      "t23"."l_comment",
      "t23"."s_suppkey",
      "t23"."s_name",
      "t23"."s_address",
      "t23"."s_nationkey",
      "t23"."s_phone",
      "t23"."s_acctbal",
      "t23"."s_comment",
      "t23"."n_nationkey",
      "t23"."n_name",
      "t23"."n_regionkey",
      "t23"."n_comment",
      "t23"."r_regionkey",
      "t23"."r_name",
      "t23"."r_comment"
    FROM (
      SELECT
        "t14"."c_custkey",
        "t14"."c_name",
        "t14"."c_address",
        "t14"."c_nationkey",
        "t14"."c_phone",
        "t14"."c_acctbal",
        "t14"."c_mktsegment",
        "t14"."c_comment",
        "t15"."o_orderkey",
        "t15"."o_custkey",
        "t15"."o_orderstatus",
        "t15"."o_totalprice",
        "t15"."o_orderdate",
        "t15"."o_orderpriority",
        "t15"."o_clerk",
        "t15"."o_shippriority",
        "t15"."o_comment",
        "t16"."l_orderkey",
        "t16"."l_partkey",
        "t16"."l_suppkey",
        "t16"."l_linenumber",
        "t16"."l_quantity",
        "t16"."l_extendedprice",
        "t16"."l_discount",
        "t16"."l_tax",
        "t16"."l_returnflag",
        "t16"."l_linestatus",
        "t16"."l_shipdate",
        "t16"."l_commitdate",
        "t16"."l_receiptdate",
        "t16"."l_shipinstruct",
        "t16"."l_shipmode",
        "t16"."l_comment",
        "t17"."s_suppkey",
        "t17"."s_name",
        "t17"."s_address",
        "t17"."s_nationkey",
        "t17"."s_phone",
        "t17"."s_acctbal",
        "t17"."s_comment",
        "t12"."n_nationkey",
        "t12"."n_name",
        "t12"."n_regionkey",
        "t12"."n_comment",
        "t13"."r_regionkey",
        "t13"."r_name",
        "t13"."r_comment"
      FROM (
        SELECT
          "t0"."c_custkey",
          "t0"."c_name",
          "t0"."c_address",
          "t0"."c_nationkey",
          "t0"."c_phone",
          CAST("t0"."c_acctbal" AS DECIMAL(15, 2)) AS "c_acctbal",
          "t0"."c_mktsegment",
          "t0"."c_comment"
        FROM "customer" AS "t0"
      ) AS "t14"
      INNER JOIN (
        SELECT
          "t1"."o_orderkey",
          "t1"."o_custkey",
          "t1"."o_orderstatus",
          CAST("t1"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t1"."o_orderdate",
          "t1"."o_orderpriority",
          "t1"."o_clerk",
          "t1"."o_shippriority",
          "t1"."o_comment"
        FROM "orders" AS "t1"
      ) AS "t15"
        ON "t14"."c_custkey" = "t15"."o_custkey"
      INNER JOIN (
        SELECT
          "t2"."l_orderkey",
          "t2"."l_partkey",
          "t2"."l_suppkey",
          "t2"."l_linenumber",
          CAST("t2"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
          CAST("t2"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
          CAST("t2"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
          CAST("t2"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
          "t2"."l_returnflag",
          "t2"."l_linestatus",
          "t2"."l_shipdate",
          "t2"."l_commitdate",
          "t2"."l_receiptdate",
          "t2"."l_shipinstruct",
          "t2"."l_shipmode",
          "t2"."l_comment"
        FROM "lineitem" AS "t2"
      ) AS "t16"
        ON "t16"."l_orderkey" = "t15"."o_orderkey"
      INNER JOIN (
        SELECT
          "t3"."s_suppkey",
          "t3"."s_name",
          "t3"."s_address",
          "t3"."s_nationkey",
          "t3"."s_phone",
          CAST("t3"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
          "t3"."s_comment"
        FROM "supplier" AS "t3"
      ) AS "t17"
        ON "t16"."l_suppkey" = "t17"."s_suppkey"
      INNER JOIN (
        SELECT
          "t4"."n_nationkey",
          "t4"."n_name",
          "t4"."n_regionkey",
          "t4"."n_comment"
        FROM "nation" AS "t4"
      ) AS "t12"
        ON "t14"."c_nationkey" = "t17"."s_nationkey"
        AND "t17"."s_nationkey" = "t12"."n_nationkey"
      INNER JOIN (
        SELECT
          "t5"."r_regionkey",
          "t5"."r_name",
          "t5"."r_comment"
        FROM "region" AS "t5"
      ) AS "t13"
        ON "t12"."n_regionkey" = "t13"."r_regionkey"
    ) AS "t23"
    WHERE
      "t23"."r_name" = 'ASIA'
      AND "t23"."o_orderdate" >= FROM_ISO8601_DATE('1994-01-01')
      AND "t23"."o_orderdate" < FROM_ISO8601_DATE('1995-01-01')
  ) AS "t24"
  GROUP BY
    1
) AS "t25"
ORDER BY
  "t25"."revenue" DESC