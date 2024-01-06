SELECT
  ST_ASEWKB(
    "t0"."<MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, ...>"
  ) AS "<MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, ...>"
FROM (
  SELECT
    ST_GEOMFROMTEXT(
      'MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35, 30 20, 20 15, 20 25, 30 20, 20 35)))'
    ) AS "<MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, ...>"
) AS "t0"