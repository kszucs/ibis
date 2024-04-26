WITH [t1] AS (
  SELECT
    *
  FROM [leaf] AS [t0]
  WHERE
    (1 = 1)
)
SELECT
  [t3].[key]
FROM [t1] AS [t3]
INNER JOIN (
  SELECT
    [t2].[key]
  FROM [t1] AS [t2]
) AS [t5]
  ON [t3].[key] = [t5].[key]
INNER JOIN (
  SELECT
    [t3].[key]
  FROM [t1] AS [t3]
  INNER JOIN (
    SELECT
      [t2].[key]
    FROM [t1] AS [t2]
  ) AS [t5]
    ON [t3].[key] = [t5].[key]
) AS [t7]
  ON [t3].[key] = [t7].[key]