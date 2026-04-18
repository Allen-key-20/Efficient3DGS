import subprocess

model = "bicycle"

with open(rf"./output/{model}.txt", "a", encoding="UTF-8") as rec:
    rec.write(f"{model}\n")

command1 = f'python train.py -s ./data/{model} -m ./output/{model} --eval --quiet '
result = subprocess.run(command1, shell=True)
if result.returncode != 0:
    exit(1)

command2 = f'python render.py -m ./output/{model} --skip_train --quiet'
subprocess.run(command2, shell=True)

command3 = f'python metrics.py -m ./output/{model}'
result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')

with open(rf"./output/{model}.txt", "a", encoding="UTF-8") as rec:
    rec.write(result.stdout)
print("============down===========")

