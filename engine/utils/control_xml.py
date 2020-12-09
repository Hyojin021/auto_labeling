import xml.etree.ElementTree as ET
import os

def create_label_map(xml_dir, label_map_path):

    xml_paths = [os.path.join(xml_dir, f.name) for f in os.scandir(xml_dir)]

    labels = []
    for xml_path in xml_paths:
        label = get_category(xml_path)
        labels += label

    labels = list(set(labels))
    label_map = []

    if not os.path.isfile(label_map_path):
        with open(label_map_path, 'w', encoding='utf-8') as f:
            for i, label in enumerate(labels):
                    f.write(label + '\n')

    else:
        with open(label_map_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                label_map.append(line[:-1])

            add_labels = list(set(labels).difference(label_map))

        with open(label_map_path, 'a+', encoding='utf-8') as f:
            for i, add_label in enumerate(add_labels):
                f.write(add_label + '\n')



def get_category(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    names = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        names.append(name)
    return names


def save_xml(bboxs, classes, heigth, width, fname, savefilepath):
    annotation = ET.Element('annotation')  # root
    ET.SubElement(annotation, 'folder').text = 'Unkown'
    ET.SubElement(annotation, 'filename').text = str(fname)
    ET.SubElement(annotation, 'path').text = 'Unkown'
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unkown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(heigth)
    ET.SubElement(size, 'depth').text = str(3)
    ET.SubElement(annotation, 'segmented').text = str(0)
    for i in range(len(bboxs)):
        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = str(classes[i])
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = str(0)
        ET.SubElement(object, 'difficult').text = str(0)
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(bboxs[i][0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(bboxs[i][1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(bboxs[i][2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(bboxs[i][3]))

    indent(annotation)
    tree = ET.ElementTree(annotation)

    tree.write(savefilepath)

    return tree


def indent(elem, level=0):
    i = "\n" + level * " "

    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + " "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i