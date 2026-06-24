import numpy as np
from vasc_pytorch import vasc
from helpers_pytorch import clustering, measure, print_2D
from config_pytorch import config

if __name__ == '__main__':
    DATASET = 'data/biase'  
    PREFIX = 'biase'  
    
    filename = DATASET + '.txt'
    data = open(filename)
    head = data.readline().rstrip().split()
    
    label_file = open(DATASET + '_label.txt')
    label_dict = {}
    for line in label_file:
        temp = line.rstrip().split()
        label_dict[temp[0]] = temp[1]
    label_file.close()
    
    label = []
    for c in head:
        if c in label_dict.keys():
            label.append(label_dict[c])
        else:
            print(c)
    
    label_set = []
    for c in label:
        if c not in label_set:
            label_set.append(c)
    name_map = {value: idx for idx, value in enumerate(label_set)}
    id_map = {idx: value for idx, value in enumerate(label_set)}
    label = np.asarray([name_map[name] for name in label])
    
    expr = []
    for line in data:
        temp = line.rstrip().split()[1:]
        temp = [float(x) for x in temp]
        expr.append(temp)
    
    expr = np.asarray(expr).T
    n_cell, _ = expr.shape
    batch_size = config['batch_size'] if n_cell > 150 else 32
    
    for i in range(1):
        print("Iteration:" + str(i))
        res = vasc(expr, epoch=config['epoch'], var=False,
                   latent=config['latent'],
                   annealing=False,
                   batch_size=batch_size,
                   prefix=PREFIX,
                   label=label,
                   scale=config['scale'],
                   patience=config['patience']
                   )
        
        print(res.shape)
        k = len(np.unique(label))
        cl, _ = clustering(res, k=k)
        dm = measure(cl, label)
        
        fig = print_2D(points=res, label=label, id_map=id_map)
