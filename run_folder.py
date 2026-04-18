import os
import subprocess
import time

m360 = 1
tat = 1
db = 1

if m360:
    if not os.path.exists(f"./output/bicycle"):
        with open(rf"./output/bicycle.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"bicycle\n")
        command1 = (f'python train.py -s ./data/m360/bicycle -m ./output/bicycle  --eval '
                    f' --imp_thresh 30  --grad_abs_thresh 0.0012')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/bicycle --skip_train  --quiet'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/bicycle'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/bicycle.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/flowers"):
        with open(rf"./output/flowers.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"flowers\n")
        command1 = (f'python train.py -s ./data/m360/flowers -m ./output/flowers  --eval '
                    f' --imp_thresh 30   --dense 0.005 --grad_abs_thresh 0.0015')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/flowers --skip_train  --quiet'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/flowers'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/flowers.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/garden"):
        with open(rf"./output/garden.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"garden\n")
        command1 = (f'python train.py -s ./data/m360/garden -m ./output/garden --eval '
                    f'  --imp_thresh 30  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/garden --skip_train  --quiet'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/garden'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/garden.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/stump"):
        with open(rf"./output/stump.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"stump\n")
        command1 = (f'python train.py -s ./data/m360/stump -m ./output/stump  --eval '
                    f' --imp_thresh 30   --dense 0.004 --grad_abs_thresh 0.0015')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/stump --skip_train  --quiet'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/stump'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/stump.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/treehill"):
        with open(rf"./output/treehill.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"treehill\n")
        command1 = (f'python train.py -s ./data/m360/treehill -m ./output/treehill  --eval '
                    f' --imp_thresh 30  --dense 0.01 --grad_abs_thresh 0.002')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/treehill --skip_train  --quiet'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/treehill'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/treehill.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/room"):
        with open(rf"./output/room.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"room\n")
        command1 = (f'python train.py -s ./data/m360/room -m ./output/room  --eval '
                    f' --imp_thresh 30   --highfeature_lr 0.02 --grad_abs_thresh 0.0008')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/room --skip_train  --quiet'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/room'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/room.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/counter"):
        with open(rf"./output/counter.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"counter\n")
        command1 = (f'python train.py -s ./data/m360/counter -m ./output/counter  --eval '
                    f' --imp_thresh 30   --highfeature_lr 0.02 --grad_abs_thresh 0.0008 ')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/counter --skip_train  --quiet'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/counter'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/counter.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/kitchen"):
        with open(rf"./output/kitchen.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"kitchen\n")
        command1 = (f'python train.py -s ./data/m360/kitchen -m ./output/kitchen  --eval '
                    f' --imp_thresh 30   --highfeature_lr 0.02 --grad_abs_thresh 0.0006')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/kitchen --skip_train  --quiet'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/kitchen'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/kitchen.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/bonsai"):
        with open(rf"./output/bonsai.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"bonsai\n")
        command1 = (f'python train.py -s ./data/m360/bonsai -m ./output/bonsai  --eval '
                    f' --imp_thresh 30   --highfeature_lr 0.02 --grad_abs_thresh 0.0006')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/bonsai --skip_train --quiet'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/bonsai'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/bonsai.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

if tat:
    if not os.path.exists(f"./output/truck"):
        with open(rf"./output/truck.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"truck\n")
        command1 = (f'python train.py -s ./data/tat/truck -m ./output/truck  --eval '
                    f' --imp_thresh 30   --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 ')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/truck --skip_train --quiet --mult 0.7'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/truck'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/truck.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/train"):
        with open(rf"./output/train.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"train\n")
        command1 = (f'python train.py -s ./data/tat/train -m ./output/train  --eval '
                    f' --imp_thresh 30  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7 ')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/train --skip_train --quiet --mult 0.7'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/train'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/train.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

if db:
    if not os.path.exists(f"./output/playroom"):
        with open(rf"./output/playroom.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"playroom\n")
        command1 = (f'python train.py -s ./data/db/playroom -m ./output/playroom  --eval '
                    f' --imp_thresh 30  --highfeature_lr 0.0015 --dense 0.003 --mult 0.7')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/playroom --skip_train --quiet --mult 0.7'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/playroom'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/playroom.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)

    if not os.path.exists(f"./output/drjohnson"):
        with open(rf"./output/drjohnson.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"drjohnson\n")
        command1 = (f'python train.py -s ./data/db/drjohnson -m ./output/drjohnson  --eval '
                    f' --imp_thresh 60 --highfeature_lr 0.0025 --grad_abs_thresh 0.0012 --dense 0.013 --mult 0.7')
        subprocess.run(command1, shell=True)
        command2 = f'python render.py -m ./output/drjohnson --skip_train --quiet --mult 0.7'
        subprocess.run(command2, shell=True)
        command3 = f'python metrics.py -m ./output/drjohnson'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./output/drjohnson.txt", "a", encoding="UTF-8") as rec:
            rec.write(result.stdout)
        print("============down===========")
        time.sleep(60)
