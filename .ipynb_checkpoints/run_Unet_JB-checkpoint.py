import subprocess
from datetime import datetime

user = 'JB'
model = 'newUnet'
batch = 12
epoch = 30
lrs = [1e-3]”
base_exp_name = 'newUnet_test'

for i in range(len(lrs)):
    curtime = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    exp_num = '0'+str(i+1) if i <= 8 else str(i+1)
    exp_name = base_exp_name+exp_num
    lr = lrs[i]
    command = f'python ./train.py -u {user} -x {exp_name} -n {model} -e {epoch} -b {batch} -l {lr}'
    print(command)
#     result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if result.stderr != '':
#         print("ERROR OCCURED")
#         print(result.stderr.decode('utf-8'))
#         break
    
#     with open(f'./logs/{user}/base_exp_name_{curtime}.txt', 'w') as f:
#         f.write(result.stdout.decode('utf-8'))
#     break 