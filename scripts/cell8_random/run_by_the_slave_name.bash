slave222="3 6 11 12 14 15 16 18 22"
slave224="23 28 29 31 32 37 44 46 50"
slave225="54 151 182 188 38 307 300 458 469"
slave226="1049 1050 1053 1063 1067 1068 1590 1510 1489"
slave227="1494 1497 1501 1480 1485 1486 1487 1468 1475"
slave228="1462 1464 4534 6332 4538 1478 23381 40499 40670 40701"
slavefuck="40668 40966 40982 40994 40983 40975 40984 40979 41027"



cd ../..
host_name=$(hostname)
for d in ${!host_name}; do
  nohup python -m benchmarks.bdas_cell -s DAG --strategy=random --max_cell=8 -d $d >${d}.txt 2>&1 &
done

logs_dir="cell_8_random"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
