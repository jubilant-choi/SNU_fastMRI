import subprocess
from datetime import datetime

model = 'VarNet'
lrs = [1e-3,1e-4,1e-5,1e-6]
base_exp_name = 'VarNet_test'

for i in range(len(lrs)):
    lr = lrs[i]
    curtime = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    exp_name = base_exp_name+str(lr)
    command = f'python ./train.py -u JB -x {exp_name} -n {model} -b 1 -e 3 -l {lr} -s P --input-key kspace --cascade 4'
    print(command)
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stderr != '':
        print("ERROR OCCURED")
        print(result.stderr.decode('utf-8'))
        break
    
    with open(f'./logs/{user}/base_exp_name_{curtime}.txt', 'w') as f:
        f.write(result.stdout.decode('utf-8'))
