
"""
Usage:
  train.py [--save_filename=<name>] \
  [--num_epochs=<n_epoch>] \
  [--initial_learning_rate=<lr>] [--learning_rate_decay=<lr_decay>]\
  [--layer_sizes=<str>] \
  [--cost_type=<ctype>] \
  [--dropout_rate=<rate>] [--lamb=<lamb>] [--epsilon=<ep>][--norm_constraint=<nc>][--num_power_iter=<npi>] \
  [--validation] [--num_validation_samples=<N>] \
  [--seed=<N>]
  train.py -h | --help

Options:
  -h --help                                 Show this screen.
  --save_filename=<name>                    [default: trained_model]
  --num_epochs=<n_ep>                       num_epochs [default: 100].
  --initial_learning_rate=<lr>              initial_learning_rate [default: 0.01].
  --learning_rate_decay=<lr_decay>          learning_rate_decay [default: 1.0].
  --layer_sizes=<str>                       layer_sizes [default: 300-200-200-2]
  --cost_type=<ctype>                       cost_type [default: MLE].
  --lamb=<lamb>                             [default: 1.0].
  --epsilon=<ep>                            [default: 2.1].
  --norm_constraint=<nc>                    [default: L2].
  --num_power_iter=<npi>                    [default: 1].
  --seed=<N>                                [default: 1].
"""

from docopt import docopt

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

from vat.source import optimizers
from vat.source import costs


import os
import errno
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

from vat.models.fnn import FNN
import vat.source.layers as L

class FNN_sentiment(FNN):

    def __init__(self,layer_sizes):
        self.linear_layers = []
        self.bn_layers = []
        self.act_layers = []
        self.params = []
        for m,n in zip(layer_sizes[:-1],layer_sizes[1:]):
            l = L.Linear(size=(m,n))
            bn = L.BatchNormalization(size=(n))
            self.linear_layers.append(l)
            self.bn_layers.append(bn)
            self.params += l.params + bn.params
        for i in xrange(len(self.linear_layers)-1):
            self.act_layers.append(L.relu)
        self.act_layers.append(L.softmax)

    def forward_for_finetuning_batch_stat(self,input):
        return self.forward(input,finetune=True)

    def forward_no_update_batch_stat(self,input,train=True):
        return self.forward(input,train,False)

    def forward(self,input,train=True,update_batch_stat=True,finetune=False):
        h = input
        for l,bn,act in zip(self.linear_layers,self.bn_layers,self.act_layers):
            h = l(h)
            h = bn(h,train=train,update_batch_stat=update_batch_stat,finetune=finetune)
            h = act(h)
        return h


def train(args,x_train,t_train,x_test,t_test,ul_x_train=None):

    print args

    numpy.random.seed(int(args['--seed']))

    layer_sizes = [int(layer_size) for layer_size in args['--layer_sizes'].split('-')] 
    model = FNN_sentiment(layer_sizes=layer_sizes)

    x = T.matrix()
    ul_x = T.matrix()
    t = T.ivector()


    if(args['--cost_type']=='MLE'):
        cost = costs.cross_entropy_loss(x=x,t=t,forward_func=model.forward_train)
    elif(args['--cost_type']=='L2'):
        cost = costs.cross_entropy_loss(x=x,t=t,forward_func=model.forward_train) \
               + costs.weight_decay(params=model.params,coeff=float(args['--lamb']))
    elif(args['--cost_type']=='AT'):
        cost = costs.adversarial_training(x,t,model.forward_train,
                                              'CE',
                                              epsilon=float(args['--epsilon']),
                                              lamb=float(args['--lamb']),
                                              norm_constraint = args['--norm_constraint'],
                                              forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif(args['--cost_type']=='VAT'):
        cost = costs.virtual_adversarial_training(x,t,model.forward_train,
                                              'CE',
                                              epsilon=float(args['--epsilon']),
                                              lamb=float(args['--lamb']),
                                              norm_constraint = args['--norm_constraint'],
                                              num_power_iter = int(args['--num_power_iter']),
                                              x_for_generating_adversarial_examples=ul_x,
                                              forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif(args['--cost_type']=='VAT_finite_diff'):
        cost = costs.virtual_adversarial_training_finite_diff(x,t,model.forward_train,
                                              'CE',
                                              epsilon=float(args['--epsilon']),
                                              lamb=float(args['--lamb']),
                                              norm_constraint = args['--norm_constraint'],
                                              num_power_iter = int(args['--num_power_iter']),
                                              x_for_generating_adversarial_examples=ul_x,
					      unchain_y = False,
                                              forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    nll = costs.cross_entropy_loss(x=x,t=t,forward_func=model.forward_test)
    error = costs.error(x=x,t=t,forward_func=model.forward_test)

    optimizer = optimizers.ADAM(cost=cost,params=model.params,alpha=float(args['--initial_learning_rate']))


    f_train = theano.function(inputs=[], outputs=cost, updates=optimizer.updates,
                              givens={
                                  x:x_train,
                                  t:t_train,
                                  ul_x:ul_x_train},on_unused_input='warn')
    f_nll_train = theano.function(inputs=[], outputs=nll,
                              givens={
                                  x:x_train,
                                  t:t_train})
    f_nll_test = theano.function(inputs=[], outputs=nll,
                              givens={
                                  x:x_test,
                                  t:t_test})

    f_error_train = theano.function(inputs=[], outputs=error,
                              givens={
                                  x:x_train,
                                  t:t_train})
    f_error_test = theano.function(inputs=[], outputs=error,
                              givens={
                                  x:x_test,
                                  t:t_test})

    f_lr_decay = theano.function(inputs=[],outputs=optimizer.alpha,
                                 updates={optimizer.alpha:theano.shared(numpy.array(args['--learning_rate_decay']).astype(theano.config.floatX))*optimizer.alpha})


    statuses = {}
    statuses['nll_train'] = []
    statuses['error_train'] = []
    statuses['nll_test'] = []
    statuses['error_test'] = []

    n_train = numpy.asarray(x_train.get_value().shape[0],theano.config.floatX)
    n_test = numpy.asarray(x_test.get_value().shape[0],theano.config.floatX)


    statuses['nll_train'].append(f_nll_train())
    statuses['error_train'].append(f_error_train()/n_train)
    statuses['nll_test'].append(f_nll_test())
    statuses['error_test'].append(f_error_test()/n_test)

    print "[Epoch]",str(-1)
    print  "nll_train : " , statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
            "nll_test : " , statuses['nll_test'][-1],  "error_test : ", statuses['error_test'][-1]

    print "training..."

    make_sure_path_exists("./trained_model")

    for epoch in xrange(int(args['--num_epochs'])):
        cPickle.dump((statuses,args),open('./trained_model/'+'tmp-' + args['--save_filename'],'wb'),cPickle.HIGHEST_PROTOCOL)

        ### update parameters ###
        f_train() 
        #########################


        statuses['nll_train'].append(f_nll_train())
        statuses['error_train'].append(f_error_train()/n_train)
        statuses['nll_test'].append(f_nll_test())
        statuses['error_test'].append(f_error_test()/n_test)
        print "[Epoch]",str(epoch)
        print  "nll_train : " , statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
                "nll_test : " , statuses['nll_test'][-1],  "error_test : ", statuses['error_test'][-1]

        f_lr_decay()

    return f_error_train()/n_train, f_error_test()/n_test
    #make_sure_path_exists("./trained_model")
    #cPickle.dump((model,statuses,args),open('./trained_model/'+args['--save_filename'],'wb'),cPickle.HIGHEST_PROTOCOL)


import numpy
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator

"""
class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization

    def fit(self, _X, y=None):
        X = _X.copy()
        self.mean_ = numpy.mean(X, axis=0)
        X -= self.mean_
        sigma = numpy.dot(X.T,X)  / X.shape[0] + self.regularization*numpy.eye(X.shape[1])
        U, S, V = linalg.svd(sigma)
        tmp = numpy.dot(U, numpy.diag(1/numpy.sqrt(S)))
        self.components_ = numpy.dot(tmp, U.T)
        return self

    def transform(self, X):
        X_transformed = X - self.mean_
        X_transformed = numpy.dot(X_transformed, self.components_.T)
        return X_transformed
"""

if __name__=='__main__':
    args = docopt(__doc__)
    x,y = cPickle.load(open("../dataset/training_dataset/dataset.pkl"))

    l_x = x[numpy.where(y != 0.5)[0]].astype(theano.config.floatX)
    l_y = y[numpy.where(y != 0.5)[0]].astype("int32")
    ul_x = x[numpy.where(y == 0.5)[0]].astype(theano.config.floatX)

    num_samples = l_x.shape[0]
    num_groups = 10
    print "chance level :" + str(1-numpy.mean(l_y))

    accs = numpy.zeros(num_groups)
    for i in xrange(num_groups):
        valid_index = numpy.arange(int(i*num_samples/num_groups),int((i+1)*num_samples/num_groups))
        train_index =numpy.delete(numpy.arange(num_samples), valid_index)

        x_train = theano.shared(l_x[train_index])
        t_train = theano.shared(l_y[train_index])
        x_valid = theano.shared(l_x[valid_index])
        t_valid = theano.shared(l_y[valid_index])

        ul_x_train = theano.shared(numpy.concatenate((l_x[train_index],ul_x),axis=0))

        error_train, error_valid = train(args,x_train,t_train,x_valid,t_valid,x_train)
        accs[i] = 1-error_valid

    print "valid error : " + str(accs.mean()) + "%"
    cPickle.dump(accs.mean(),open("Valid_accuracy_" + args["--cost_type"] + "_epsilon" + args["--epsilon"] + ".pkl","wb"))
