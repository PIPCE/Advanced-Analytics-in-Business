#!/usr/bin/env sh

echo I assume you have Python and Jupyter installed

CURRENT_HOME="$( cd "$(dirname "$0")" ; pwd -P )"
export SPARK_HOME=$CURRENT_HOME/spark-3.1.1-bin-hadoop2.7
export JAVA_HOME=$CURRENT_HOME/jre1.8.0_241_mac
export PYSPARK=$SPARK_HOME/bin/pyspark

export PYSPARK_DRIVER_PYTHON="jupyter" 
export PYSPARK_DRIVER_PYTHON_OPTS="notebook" 
export PYSPARK_PYTHON=python

echo Ready to start
echo $PYSPARK

$PYSPARK
