cd ../../..

slaves="212 213 214 215 216 217 219 220 221 222 223 224 225 226 227 228 229"

for slave in $slaves; do
  echo "begin to extract files..."
  ssh slave$slave "cd ~/qijiahao;rm -rf qi;tar -zxvf qi.tar.gz"
done
