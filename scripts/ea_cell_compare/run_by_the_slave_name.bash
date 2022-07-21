source activate bdas
cd ../..
host_name=$(hostname)
slave222="6 1590 40597"
for d in ${!host_name}; do
  nohup python -m benchmarks.ea_iter -s plain  -d $d >${d}.txt 2>&1 &
  nohup python -m benchmarks.ea_iter -s plain  -d $d >${d}.txt 2>&1 &
  nohup python -m benchmarks.ea_iter -s plain  -d $d >${d}.txt 2>&1 &
  nohup python -m benchmarks.ea_iter -s plain  -d $d >${d}.txt 2>&1 &
  nohup python -m benchmarks.ea_iter -s plain  -d $d >${d}.txt 2>&1 &
  nohup python -m benchmarks.ea_iter -s plain  -d $d >${d}.txt 2>&1 &
  nohup python -m benchmarks.ea_iter -s plain  -d $d >${d}.txt 2>&1 &
done

logs_dir="ea_plain_100"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
