import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import codecs
import torch

def log_excel(excel_path, args):
    excel_writer = pd.ExcelWriter(excel_path)

    for (sheet_name, data) in args:
        result_excel = pd.DataFrame.from_dict(data=data)
        result_excel.to_excel(excel_writer, sheet_name)

def log_txt(txt_path,  *args):
    try:
        with codecs.open(txt_path, "a", "utf-8") as log_file:
            for line in args:
                log_file.write(line + "\n")
    except:
        print("Fail!")

def plot_graph_save(save_path, x_label, y_label, title, dict_data):
    figure = plt.figure(dpi=300)
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
    plt.savefig(os.path.join(file_path, "confusion_matrix.png"), dpi=300)
    plt.close()

def weight_histogram(model):
    model.parameters()