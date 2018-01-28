#!/usr/bin/env bash

export IBIS_TEST_IMPALA_HOST=localhost
export IBIS_TEST_IMPALA_PORT=21050
export IBIS_TEST_NN_HOST=localhost
export IBIS_TEST_WEBHDFS_PORT=50070
export IBIS_TEST_WEBHDFS_USER=ubuntu
export IBIS_TEST_DATA_DIRECTORY=/tmp/ibis-testing-data

CWD=$(dirname $0)

python $CWD/datamgr.py download
python $CWD/datamgr.py sqlite
python $CWD/datamgr.py postgres
python $CWD/datamgr.py clickhouse
python $CWD/impalamgr.py load --data --no-udf --data-dir $IBIS_TEST_DATA_DIRECTORY

