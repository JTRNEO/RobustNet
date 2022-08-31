import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import pickle
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


def histogram(metric, plate, domains, num_classes):
    x = np.arange(len(domains))
    total_width, n = 0.8, num_classes
    width = total_width/n

    x = x - (total_width - width)/2
    print(x)    

    for k,v in plate.items():
        plt.bar(x + v*width, metric[v], width=width, label=k)
        
    plt.xticks(x, domains, fontsize=5, rotation=45)
    plt.legend()
    plt.show()
    
def plot_embedding(data, label, names, colors, title):
    
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    ids=range(len(names))
    fig = plt.figure(figsize=(6, 5))
    for i, c, name in zip(ids, colors, names):
        plt.scatter(data[label == i, 0], data[label == i, 1], c=c, label=name)
    plt.xticks()
    plt.yticks()
    plt.title(title, fontsize=14)
    return fig

# 主函数，执行t-SNE降维
def main():
    root = './data/OEM/continent_trainval/'
    domains = os.listdir(root)
    num_classes = 8

    plate = {"Bareland": 0,
        "Grass": 1,
        "Pavement": 2,
        "Road": 3,
        "Tree": 4,
        "Water": 5,
        "Cropland": 6,
        "buildings": 7}
    
    points = []
    labels = []
    
    for domain in domains:
        globals()[domain] = []
    for i,domain in tqdm(enumerate(domains)):
        dataset = os.path.join(root, domain, 'labels', 'val')
        fre = np.zeros((1, num_classes), dtype=np.int64)
        for mask_path in os.listdir(dataset):
            mask = Image.open(os.path.join(dataset, mask_path))
            mask = np.array(mask).flatten()
            fre += np.bincount(mask, minlength=num_classes+1)[1:]
            prob = fre/fre.sum()
            points.append(prob)
            labels.append(i)
        eval(domain).append(fre)


    metric = np.concatenate((eval(domains[0])[0]/eval(domains[0])[0].sum(), 
                            eval(domains[1])[0]/eval(domains[1])[0].sum(), 
                            eval(domains[2])[0]/eval(domains[2])[0].sum(), 
                            eval(domains[3])[0]/eval(domains[3])[0].sum(), 
                            eval(domains[4])[0]/eval(domains[4])[0].sum(), 
                            eval(domains[5])[0]/eval(domains[5])[0].sum(), 
                            eval(domains[6])[0]/eval(domains[6])[0].sum()), axis=0).T
    

    data = np.squeeze(np.array(points))
    label = np.squeeze(np.array(labels))
    colors='r','g','b','c','m','y','k'
    # print('Starting plot Histogram...')
    # histogram(metric=metric, plate=plate, domains=domains, num_classes=num_classes)
    
    print('Starting compute t-SNE Embedding...')
    
    ts = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    reslut = ts.fit_transform(data)
    # 调用函数，绘制图像
    fig = plot_embedding(reslut, label, domains, colors, 't-SNE Embedding of content')
    # 显示图像
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
