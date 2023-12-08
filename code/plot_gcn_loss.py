import numpy as np
import seaborn as sns
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(loss_df, title):
    sns.set_style('darkgrid')
    sns.set_context('notebook')
    sns.lineplot(data = loss_df)
    sns.despine()
    sns.set_context('talk')
    sns.set_style('white')
    plt.title(title)
    plt.show()

base_loss = pk.load(open('/ix/djishnu/Hanxi/PGM_Project/GCN_base_loss.pkl', 'rb'))
new_loss = pk.load(open('/ix/djishnu/Hanxi/PGM_Project/GCN_word_loss.pkl', 'rb'))
wp_loss = pk.load(open('/ix/djishnu/Hanxi/PGM_Project/GCN_wp_loss.pkl', 'rb'))

losses = [base_loss, new_loss, wp_loss]
loss_df = pd.DataFrame(losses).transpose()
loss_df.columns = ['word only', 'word + neighbors', 'word + position']
plot_loss(loss_df, 'GCN Losses')


