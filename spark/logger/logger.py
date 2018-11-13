import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import codecs
import ntpath
from ..analyze import ActivationMap, trainable_parameters

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def save_excel(excel_path, args):
    """
    logger function to write input data to output excel file.
    :param excel_path: save excel file path
    :param args: ( "sheet name", dictionary)
    :return: None
    """
    excel_writer = pd.ExcelWriter(excel_path)

    for (sheet_name, data) in args:
        result_excel = pd.DataFrame.from_dict(data=data)
        result_excel.to_excel(excel_writer, sheet_name)

def save_txt(txt_path,  *args):
    with codecs.open(txt_path, "a", "utf-8") as log_file:
        for line in args:
            log_file.write(line + "\n")

def save_2D_graph(dict_data, x_label, y_label, title, save_path ):
    figure = plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    for model_name in dict_data.keys():
        plt.plot([x for x in range(len(dict_data[model_name]))], dict_data[model_name], label=model_name)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def logit_to_excel(paths, logits, label, file_path):
    paths_excel = pd.DataFrame(paths, columns=["file_path"])
    logits_excel = pd.DataFrame(logits,
                                columns=["class_{}_logit".format(x) for x in range(logits.shape[1])])
    labels = pd.DataFrame(label, columns=["label"])
    all = pd.concat([paths_excel, logits_excel, labels], axis=1)
    all.to_csv(os.path.join(file_path, "extracted_logits.csv"))

def features_to_excel(paths, features, label, file_path):
    paths_excel = pd.DataFrame(paths,columns=["file_path"])
    features_excel = pd.DataFrame(features,
                                columns=["{}th_feat".format(x) for x in range(features.shape[1])])
    labels = pd.DataFrame(label, columns=["label"])
    all = pd.concat([paths_excel, features_excel, labels], axis=1)
    all.to_csv(os.path.join(file_path, "extracted_features.csv"))

def cm_to_excel(cm, class_label, file_path):
    features_excel = pd.DataFrame(cm, index=class_label,
                                  columns=class_label)
    features_excel.to_csv(os.path.join(file_path, "confusion_matrix.csv"))

def plot_confusion_matrix(cm, classes,file_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "confusion_matrix.png"))
    plt.close()

def draw_weight_histogram(model, file_path):
    hist = trainable_parameters(model)
    hist = [weight.detach().numpy() for weight in hist]

    plt.hist(hist)
    plt.xlabel("weight")
    plt.savefig(os.path.join(file_path, "weight_histogram.png"))
    plt.close()

def gridimages(path, images, cols=1, subtitles=None, title=None):
    assert ((subtitles is None) or (len(images) == len(subtitles)))
    n_images = len(images)
    if subtitles is None: subtitles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    plt.axis("off")
    plt.title(title)
    for n, (image, title) in enumerate(zip(images, subtitles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1, xbound=0)
        if image.ndim == 2:
            plt.gray()
        plt.axis("off")
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images/6)
    plt.savefig(path)
    plt.close()

def sample_one_image(path, num_max = 10):
    sample_dict = {}
    count = 0
    for root, dirs, files in os.walk(path):
        if count > num_max:
            break
        for dir in dirs:
            current_dir = os.path.join(root, dir)
            for _, _, files in os.walk(current_dir):
                sample_dict.setdefault(dir, os.path.join(current_dir, files[0]))
                break
        break
    return sample_dict

