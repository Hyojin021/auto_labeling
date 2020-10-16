import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET


def load_xml(anno_path):

    tree = ET.parse(anno_path)
    root = tree.getroot()

    target = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        target.append(name)
    return target


def main(args=None):

    df = pd.DataFrame(columns=['class'])
    anno_paths = [os.path.join(args.anno_dir, f.name) for f in os.scandir(args.anno_dir)]

    for anno_path in anno_paths:
        target = load_xml(anno_path)
        for label in target:
            df = df.append({'class': label}, ignore_index=True)
    sns.countplot(x='class', data=df)
    # sns.barplot(x='class', data=df)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Histo-category')
    parser.add_argument('--anno_dir', type=str, default='/media/jsk/data/namwon/defect/Not_Print_img/crop/anno')
    args = parser.parse_args()

    main(args)