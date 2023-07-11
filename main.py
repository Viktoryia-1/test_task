import seaborn as sns
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn_evaluation import plot
from scipy import stats
from statannotations.Annotator import Annotator
import os
import datetime

class DrawPlots:
  def __init__(self, link, hide=False):
    self.data = pd.DataFrame(requests.get(link).json())
    self.path = os.getcwd() + '/plots_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self.hide = hide

  def compare_target(self):

    # Look on classes balance in true and predicted values
    true_labels = self.data.gt_corners
    pred_labels = self.data.rb_corners

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('True values and predictions comparison')
    colors = {'corners_4':'lightblue', 'corners_6':'coral','corners_8':'fuchsia', 'corners_10':'yellow'}
    ax1.bar(true_labels.value_counts().index, true_labels.value_counts(),color=colors.values(), label=colors.keys())
    ax1.legend()
    ax2.bar(pred_labels.value_counts().index, pred_labels.value_counts(), color=colors.values(), label=colors.keys());
    ax2.legend()
    if not os.path.exists(self.path):
      os.mkdir(self.path)
      fig.savefig(self.path + '/compare_target.png')
    else:
      fig.savefig(self.path + '/compare_target.png')
    if self.hide:
      plt.close(fig)

  def plot_cls_report(self):

    true_labels = self.data.gt_corners
    pred_labels = self.data.rb_corners

    labels = sorted(set(true_labels))
    labels = [f'{str(int(i))} corners' for i in labels]
    plot.ClassificationReport.from_raw_data(true_labels, pred_labels, target_names=labels)
    if not os.path.exists(self.path):
      os.mkdir(self.path)
      plt.savefig(self.path + '/classification_report.png')
    else:
      plt.savefig(self.path + '/classification_report.png')
    if self.hide:
      plt.close()

  def plot_conf_matrix(self):
    true_labels = self.data.gt_corners
    pred_labels = self.data.rb_corners

    fig, ax = plt.subplots(figsize=(8,7))
    ax.set_title('Confusion matrix')
    labels = sorted(set(true_labels))
    labels = [f'{str(int(i))} corners' for i in labels]
    ConfusionMatrixDisplay.from_predictions(true_labels, pred_labels, ax=ax, display_labels=labels, cmap=plt.cm.Blues)
    if not os.path.exists(self.path):
      os.mkdir(self.path)
      plt.savefig(self.path + '/confusion_matrix.png')
    else:
      plt.savefig(self.path + '/confusion_matrix.png')
    if self.hide:
      plt.close(fig)

  def plot_corr_matrix(self):
    corr = self.data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title('Correlation matrix')
    sns.heatmap(corr, annot=True, ax=ax, cmap=plt.cm.Blues)
    if not os.path.exists(self.path):
      os.mkdir(self.path)
      fig.savefig(self.path + '/correlation_matrix.png')
    else:
      fig.savefig(self.path + '/correlation_matrix.png')

    if self.hide:
      plt.close(fig)

  def plot_distribution(self):

    # Check distribution of all features
    plt.figure(figsize=(16, 18))
    for ind, el in enumerate(self.data.columns):
      corner_4 = self.data[self.data.gt_corners == 4.0]
      corner_6 = self.data[self.data.gt_corners == 6.0]
      corner_8 = self.data[self.data.gt_corners == 8.0]
      plt.subplot(6, 4, ind + 1)
      plt.subplots_adjust(hspace=0.4)
      plt.hist(corner_4[el], color='blue')
      plt.hist(corner_6[el], color='pink')
      plt.hist(corner_8[el], color='orange')
      plt.yscale('log')
      plt.title(el)
    if not os.path.exists(self.path):
      os.mkdir(self.path)
      plt.savefig(self.path +'/features_distribution.png')
    else:
      plt.savefig(self.path + '/features_distribution.png')
    if self.hide:
      plt.close()

  def plot_name(self):

    # Plot name more detailed (for 4 and 6 corners only 50 more often meets)
    corner_4 = self.data[self.data.gt_corners == 4.0]
    corner_6 = self.data[self.data.gt_corners == 6.0]
    corner_8 = self.data[self.data.gt_corners == 8.0]

    bar_4_name = corner_4['name'].value_counts().index[:50]
    bar_4_value = corner_4['name'].value_counts()[:50]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title('Room names distribution')
    ax.bar(bar_4_name, bar_4_value, color='lightblue', label='4 corners')
    ax.tick_params(rotation=90)

    bar_6_name = corner_6['name'].value_counts().index[:50]
    bar_6_value = corner_6['name'].value_counts()[:50]

    ax.bar(bar_6_name, bar_6_value, color='goldenrod', label='6 corners')

    bar_8_name = corner_8['name'].value_counts().index
    bar_8_value = corner_8['name'].value_counts()

    ax.bar(bar_8_name, bar_8_value, color='orchid', label='8 corners')
    ax.legend()
    if not os.path.exists(self.path):
      os.mkdir(self.path)
      fig.savefig(self.path + '/name_distribution.png')
    else:
      fig.savefig(self.path + '/name_distribution.png')
    if self.hide:
      plt.close(fig)

  def show_mann_statistic(self, num_iterations=20, sample_size=21):
    # Join received statistics in one plot

    columns = self.data.columns.drop(['name', 'gt_corners', 'rb_corners']).to_list()
    fig = plt.figure(figsize=(14, 14))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.subplots_adjust(hspace=0.4)
        ax = self.show_boxplot(feature=columns[i],num_iterations=num_iterations, sample_size=sample_size)

    if not os.path.exists(self.path):
      os.mkdir(self.path)
      fig.savefig(self.path + '/mann_statistic.png')
    else:
      fig.savefig(self.path + '/mann_statistic.png')
    if self.hide:
      plt.close(fig)

  def show_boxplot(self, feature, num_iterations, sample_size):

    ''' In this case we have to use nonparametric tests,
     I choose Mann-Whitney criterion. Because of we have class imbalanced data I
     decided to analyze mean p-value. Hear I take a sample from biggest classes
     and calculate average p-value. It's possible to pass different options to
     num_iterations and sample_size parameters. Note that plot was created from all data,
     not sampled
     21 <= sample_size <= 160 (For Mann-Whitney criterion sample don't have to be big,
     but it isn't necessary to have them properly equal. I used values from 21 to 50) '''

    data = self.data.drop(['name', 'rb_corners'], axis=1)
    data = self.data.loc[self.data['gt_corners'] != 10.0]

    p_4_6 = 0
    p_4_8 = 0
    p_6_8 = 0

    for i in range(num_iterations):
      corner_4 = self.data[self.data['gt_corners'] == 4.0].sample(sample_size)
      corner_6 = self.data[self.data['gt_corners'] == 6.0].sample(sample_size)
      corner_8 = self.data[self.data['gt_corners'] == 8.0]
      p_4_6 += stats.mannwhitneyu(corner_4[feature], corner_6[feature]).pvalue
      p_4_8 += stats.mannwhitneyu(corner_4[feature], corner_8[feature]).pvalue
      p_6_8 += stats.mannwhitneyu(corner_6[feature], corner_8[feature]).pvalue


    p_values = [p_4_6/num_iterations, p_4_8/num_iterations, p_6_8/num_iterations]

    plotting = {
      'data':   data,
      'x':       'gt_corners',
      'y':       feature}

    ax = sns.boxplot(**plotting)
    ax.set_title(f'Statistical difference of feature {feature} \n between 4/6/8 corners group \n with num_iterations {num_iterations}')
    annot = Annotator(ax,[(4.0, 6.0), (4.0, 8.0), (6.0, 8.0)], **plotting, verbose=False)
    annot.set_custom_annotations([f'p={p_value:.3}' for p_value in p_values])
    annot.annotate()


def draw_plots(link, hide=True):
  data = DrawPlots(link)
  data.compare_target()
  data.plot_cls_report()
  data.plot_conf_matrix()
  data.plot_corr_matrix()
  data.plot_distribution()
  data.plot_name()
  data.show_mann_statistic()
  return data.path


if __name__ == '__main__':

  plots_path = draw_plots('https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json')
  print(plots_path)


