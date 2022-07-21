cd ../..

source activate bdas
# ensemble
nohup python -m benchmarks.ensemble_cv -d 181 2>&1 > ensemble_comp.log 2>&1 &

# mlp
nohup python -m benchmarks.mlp_64_32_cv -d 181 2>&1 > mlp_comp.log 2>&1 &

# ea
nohup python -m benchmarks.ea_iter_cv -s DAG  -d 181 >ea_dag_181.txt 2>&1 &
nohup python -m benchmarks.ea_iter_cv -s plain  -d 181 >ea_plain_181.txt 2>&1 &

# bdas
nohup python -m benchmarks.bdas_time_cv -s DAG -t 3600 -d 181 >bdas_plain_181.txt 2>&1 &
nohup python -m benchmarks.bdas_time_cv -s plain -t 3600 -d 181 >bdas_dag_181.txt 2>&1 &
