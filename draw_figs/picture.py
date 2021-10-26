import matplotlib.pyplot as plt

font = {'family': 'Times New Roman',
        #          'style': 'italic',
        'weight': 'normal',
        'size': 70,
        }

font2 = {'family': 'Times New Roman',
         #          'style': 'italic',
         'weight': 'bold',
         'size': 70,
         }
xtick_size = 60
ytick_size = 60
plt.figure(figsize=(50, 40))
plt.tick_params(labelsize=ytick_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.4, hspace=0.4)
plt.subplot(2, 1, 1)
name_list = ['Cora', 'Pubmed', 'Citeseer', 'Cornell', 'NBA', 'BGP', 'Electronics']
num_list1 = [69.58, 86.80, 65.72, 91.08, 69.52, 60.37, 53.40]
num_list2 = [84.29, 87.69, 74.52, 85.41, 65.40, 65.42, 69.70]
num_list3 = [75.23, 85.49, 74.60, 89.19, 66.35, 62.73, 66.24]
num_list4 = [85.67, 88.14, 78.71, 90.81, 69.52, 65.72, 76.80]
x = list(range(len(num_list1)))
total_width, n = 0.6, 3
width = total_width / n

plt.bar(x, num_list1, width=width, label='PAGG-Max', color='wheat')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='PAGG-Sum', tick_label=name_list, color='lightblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width, label='PAGG-Complete', color='salmon')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list4, width=width, label='PAGG', color='dodgerblue')
plt.legend(prop=font)
plt.xticks(fontsize=xtick_size)
# plt.xticks(range(len(max_std)))
plt.yticks(fontsize=ytick_size)
plt.xlabel('(a)Model variants of different path embedding aggregators', font2)
plt.ylabel('Accuracy', font2)
plt.ylim(50,95)

plt.subplot(2, 1, 2)
name_list = ['Cora', 'Pubmed', 'Citeseer', 'Cornell', 'NBA', 'BGP', 'Electronics']
num_list1 = [82.80, 87.75, 78.48, 90.54, 67.62, 64.79, 71.66]
num_list2 = [84.10, 87.76, 78.39, 88.92, 69.84, 64.59, 70.94]
num_list3 = [85.67, 88.14, 78.71, 90.81, 69.52, 65.72, 76.80]
num_list4 = [84.95, 87.60, 77.22, 88.38, 67.14, 64.22, 70.03]
x = list(range(len(num_list1)))

plt.bar(x, num_list1, width=width, label='1-hop', color='wheat')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='2-hops', tick_label=name_list, color='lightblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width, label='3-hops(PAGG)', color='salmon')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list4, width=width, label='4-hops', color='dodgerblue')
plt.legend(prop=font)
plt.xticks(fontsize=xtick_size)
plt.yticks(fontsize=ytick_size)
plt.xlabel('(b)Model variants of different path length', font2)
plt.ylabel('Accuracy', font2)
plt.ylim(50,95)
plt.show()
plt.savefig(fname="aggregator.eps", format="eps", pad_inches = 0, bbox_inches='tight')