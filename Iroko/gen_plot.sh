INPUT=results
OUTPUT=rate
python plot_rate.py --input $INPUT --out $OUTPUT

python plot_qlen.py --input $INPUT
