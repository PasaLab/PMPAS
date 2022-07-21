#!/bin/bash
slave210="3 6 11 12 14 15 16 18 22"
slave211="23 28 29 31 32 37 44 46 50"
slave212="54 151 182 188 38 307 300 458 469"
slave214="1049 1050 1053 1063 1067 1068 1590 1510 1489"
slave215="1494 1497 1501 1480 1485 1486 1487 1468 1475"
slave216="1462 1464 4534 6332 1461 4538 1478 23381 40499"
slave217="40668 40966 40982 40994 40983 40975 40984 40979 41027"
slave219="23517 40670 40701"

cd ../..
host_name=$(hostname)
for d in ${!host_name}; do
  nohup python -m benchmarks.bdas_time -s DAG -t 36000 -d $d >$d.txt 2>&1 &
done

logs_dir="bdas_dag_10h_logs"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
