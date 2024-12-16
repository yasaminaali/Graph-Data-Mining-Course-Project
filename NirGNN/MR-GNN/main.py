import argparse
import pickle
import time 
from utils import build_graph, Data, split_validation
from model import *



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yelpsmall', help='dataset name')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=5, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decays')
parser.add_argument('--l2', type=float, default=1e-5, help='L2 penalty')
parser.add_argument('--step', type=int, default=1, help='GNN propagation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epochs to wait before early stop')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='use a validation set')
parser.add_argument('--valid_portion', type=float, default=0.1, help='portion of training set to use as validation set')
parser.add_argument('--lambda_alpha', type=float, default=0.5, help='weight for alpha-intent (short-term intent)')
parser.add_argument('--lambda_beta', type=float, default=0.5, help='weight for beta-intent (long-term intent)')
parser.add_argument('--gamma', type=float, default=0.3, help='weight to balance cross-entropy loss and zero-shot loss')

opt = parser.parse_args()
print(opt)


def main():
    file_path_train = f'/Users/yasamin/Desktop/NirGNN/datasets/{opt.dataset}/train.txt'
    train_data_raw = pickle.load(open(file_path_train, 'rb'))
    file_path_attr = f'/Users/yasamin/Desktop/NirGNN/datasets/{opt.dataset}/all_attr.txt'
    attr_data = pickle.load(open(file_path_attr, 'rb'))
    file_path_taxo = f'/Users/yasamin/Desktop/NirGNN/datasets/{opt.dataset}/all_taxo.txt'
    taxo_data = pickle.load(open(file_path_taxo, 'rb'))
    
    # Build relationship graphs
    relationship_types = ['co-view', 'co-purchase', 'co-add-to-cart']
    relationship_graphs = build_graph(train_data_raw[0], relationship_types)

    if opt.validation:
        train_data_raw, valid_data_raw = split_validation(train_data_raw, opt.valid_portion)
        test_data_raw = valid_data_raw
    else:
        file_path_test = f'/Users/yasamin/Desktop/NirGNN/datasets/{opt.dataset}/test.txt'
        test_data_raw = pickle.load(open(file_path_test, 'rb'))
    
    # Convert datasets to Data objects with relationship graphs
    train_data = Data(train_data_raw, attr_data, relationship_graphs, shuffle=True)
    test_data = Data(test_data_raw, attr_data, relationship_graphs, shuffle=False)

    if opt.dataset == 'yelpsmall':
        n_node = 743601
    elif opt.dataset == 'mt_all' or opt.dataset == 'yoochoose1_4':
        n_node = 684
    else:
        n_node = 18889

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        precision10, mrr10, precision20, mrr20 = train_test(model, train_data, test_data, attr_data, taxo_data)
    
        flag = 0
        if precision20 >= best_result[0]:
            best_result[0] = precision20
            best_epoch[0] = epoch
            flag = 1
        if mrr20 >= best_result[1]:
            best_result[1] = mrr20
            best_epoch[1] = epoch
            flag = 1
        if precision10 >= best_result[2]:
            best_result[2] = precision10
            best_epoch[2] = epoch
            flag = 1
        if mrr10 >= best_result[3]:
            best_result[3] = mrr10
            best_epoch[3] = epoch
            flag = 1
    
        print('Best Result:')
        print('\tPrecision@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        print('\tPrecision@10:\t%.4f\tMRR@10:\t%.4f\tEpoch:\t%d,\t%d' % (best_result[2], best_result[3], best_epoch[2], best_epoch[3]))
    
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break

    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()
