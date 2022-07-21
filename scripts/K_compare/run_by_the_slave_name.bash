slave225="6 12 14 16"
slave226="22 23 28 32"
slave227="181 1590"

cd ../..
host_name=$(hostname)


for d in ${!host_name}; do
  nohup python -m benchmarks.bdas_cell_8 -K 1 -s DAG -d $d >${d}_1.txt 2>&1 &
  nohup python -m benchmarks.bdas_cell_8 -K 2 -s DAG -d $d >${d}_2.txt 2>&1 &
  nohup python -m benchmarks.bdas_cell_8 -K 4 -s DAG -d $d >${d}_4.txt 2>&1 &
  nohup python -m benchmarks.bdas_cell_8 -K 8 -s DAG  -d $d >${d}_8.txt 2>&1 &
  nohup python -m benchmarks.bdas_cell_8 -K 16 -s DAG  -d $d >${d}_16.txt 2>&1 &

done

logs_dir="bdas_K_compare"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
