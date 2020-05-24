#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[ ]:


#### Data ####

BATCH_SIZE = 64
DATASET = "CIFAR100"
NUM_CLASSES = 100

def normalize_data(X_train, X_test):
    _mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    _std = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - _mean) / _std
    X_test = (X_test - _mean) / _std
    return X_train, X_test
    
def get_tf_dataset_iter(data):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.repeat(None)    # Repeat Indefinitely
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    dataiter = iter(dataset)
    return dataiter


# In[3]:


def get_data(dataset):
    if dataset == "CIFAR100":
        data = tf.keras.datasets.cifar100.load_data()
    elif dataset == "CIFAR10":
        data = tf.keras.datasets.cifar10.load_data()
    else:
        print ("Invalid Dataset!!")
        exit(-1)

    (X_train, y_train), (X_test, y_test) = data
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    X_train, X_test = normalize_data(X_train, X_test)
    
    X_train, X_val = np.split(X_train, [int(0.98*len(X_train))])
    y_train, y_val = np.split(y_train, [int(0.98*len(y_train))])

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = get_data(DATASET)
X_train_iter = get_tf_dataset_iter(X_train)
y_train_iter = get_tf_dataset_iter(y_train)
X_val_iter = get_tf_dataset_iter(X_val)
y_val_iter = get_tf_dataset_iter(y_val)
X_test_iter = get_tf_dataset_iter(X_test)
y_test_iter = get_tf_dataset_iter(y_test)


# In[4]:


def timeme(method):
    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))

        print("Execution Time:", endTime - startTime,'ms')
        return result

    return wrapper


# In[5]:


def get_accuracy(model, Xset, yset, Xiter, yiter):
    total_batches = (Xset.shape[0] // BATCH_SIZE) + 1
    num_correct, num_samples = 0, 0
    for batch_num in range(total_batches):
        batch_X = next(Xiter)
        batch_y = next(yiter).numpy()
        scores = model(batch_X, is_training = False)
        scores = scores.numpy()
        pred_y = scores.argmax(axis=1)
        num_samples += batch_X.shape[0]
        num_correct += (pred_y == batch_y).sum()
    accuracy = float(num_correct) / num_samples
    return 100*accuracy


# In[31]:


EPOCH_ITERATIONS = 700
LEARNING_RATE = 3e-4
EXPERIMENT_ROOT = './experiment'
PRINT_FREQUENCY = 100

@timeme
# def train(CNN_class, num_epochs = 10, resume = False, plot_losses = True):
def train(model, num_epochs = 10, resume = False, plot_losses = True):
#     model = CNN_class(NUM_CLASSES)
    decay_start_iteration = 0.6 * (num_epochs * EPOCH_ITERATIONS)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE, (num_epochs * EPOCH_ITERATIONS) - decay_start_iteration, 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    writer = tf.summary.create_file_writer(EXPERIMENT_ROOT)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, EXPERIMENT_ROOT, max_to_keep=10)

    if resume:
        ckpt.restore(manager.latest_checkpoint)

    stop_training = False
    for epc in range(1, num_epochs + 1):
        print ("[Epoch {}]".format(epc))
        loss_list = []
        validation_accuracies = []
        for iter_num in range(EPOCH_ITERATIONS):
            batch_X = next(X_train_iter)
            batch_y = next(y_train_iter)
            with tf.GradientTape() as tape:
                scores = model(batch_X, is_training = True)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_y, logits=scores)
                meanloss = tf.reduce_mean(losses)
                loss_list.append(meanloss)
                lossnp = losses.numpy()

            if iter_num % PRINT_FREQUENCY == 0:

                with writer.as_default():
                    tf.summary.scalar("loss", meanloss, step=iter_num)
                    tf.summary.histogram('losses',losses,step=iter_num)

                print('iter:{:6d}, loss min|avg|max: {:.3f}|{:.3f}|{:6.3f}, '
                      .format(iter_num,
                              float(np.min(lossnp)),
                              float(np.mean(lossnp)),
                              float(np.max(lossnp))), end="")
                validation_acc = get_accuracy(model, X_val, y_val, X_val_iter, y_val_iter)
                validation_accuracies.append(validation_acc)
                print ('val acc: {:.2f} %'.format(validation_acc))
                
                if epc >= 3 and validation_acc <= 20.0:
                    print ("Circuit Breaker!! Validation accuracy too low")
                    stop_training = True
                    break

            grad = tape.gradient(meanloss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            ckpt.step.assign_add(1)
            
        manager.save()
        
        if stop_training:
            break
            
        if plot_losses:
            plt.plot(loss_list, 'r')
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(epc))
            plt.xlabel('MiniBatch number')
            plt.ylabel('MiniBatch loss')
            plt.show()
            
            plt.plot(validation_accuracies)
            plt.grid(True)
            plt.title('Epoch {} Validation Accuracies'.format(epc))
            plt.xlabel('MiniBatch number (100s)')
            plt.ylabel('Validation Accuracy')
            plt.show()
#     return model


# ### CNN1

# In[7]:


class CNN1(tf.keras.Model):
    def __init__(self, num_classes, input_shape = (32, 32, 3)):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape = input_shape, filters = 16, kernel_size = [5,5],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv1')
        self.conv2 = tf.keras.layers.Conv2D(filters = 32, kernel_size = [5,5],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv2')
        self.flatten1 = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc1')
        self.fc2 = tf.keras.layers.Dense(num_classes, name = 'fc2')

    def call(self, batch_input, is_training):
        x = self.conv1(batch_input)
        x = self.conv2(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ### CNN2

# In[8]:


class CNN2(tf.keras.Model):
    def __init__(self, num_classes, input_shape = (32, 32, 3)):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape = input_shape, filters = 16, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv1')
        self.conv2 = tf.keras.layers.Conv2D(filters = 32, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv2')
        self.flatten1 = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc1')
        self.fc2 = tf.keras.layers.Dense(num_classes, name = 'fc2')

    def call(self, batch_input, is_training):
        x = self.conv1(batch_input)
        x = self.conv2(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ### CNN3

# In[9]:


class CNN3(tf.keras.Model):
    def __init__(self, num_classes, input_shape = (32, 32, 3)):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape = input_shape, filters = 32, kernel_size = [5,5],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv1')
        self.conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = [5,5],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv2')
        self.flatten1 = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc1')
        self.fc2 = tf.keras.layers.Dense(num_classes, name = 'fc2')

    def call(self, batch_input, is_training):
        x = self.conv1(batch_input)
        x = self.conv2(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ### CNN4

# In[10]:


class CNN4(tf.keras.Model):
    def __init__(self, num_classes, input_shape = (32, 32, 3)):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape = input_shape, filters = 32, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv1')
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size = [2,2], strides = [2,2], name = 'max_pool1')
        self.conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv2')
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size = [2,2], strides = [2,2], name = 'max_pool2')
        self.flatten1 = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc1')
        self.bnorm1 = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9,epsilon=1e-5, name = 'bnorm1')
        self.fc2 = tf.keras.layers.Dense(num_classes, name = 'fc2')

    def call(self, batch_input, is_training):
        x = self.conv1(batch_input)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.bnorm1(x)
        x = self.fc2(x)
        return x


# ### CNN5

# In[11]:


class CNN5(tf.keras.Model):
    def __init__(self, num_classes, input_shape = (32, 32, 3)):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape = input_shape, filters = 32, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv1')
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size = [2,2], strides = [2,2], name = 'max_pool1')
        self.conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv2')
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size = [2,2], strides = [2,2], name = 'max_pool2')
        self.flatten1 = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc1')
        self.bnorm1 = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9,epsilon=1e-5, name = 'bnorm1')
        self.fc2 = tf.keras.layers.Dense(num_classes, name = 'fc2')

    def call(self, batch_input, is_training):
        x = self.conv1(batch_input)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.bnorm1(x, is_training)
        x = self.fc2(x)
        return x


# ### CNN6

# In[27]:


class CNN6(tf.keras.Model):
    def __init__(self, num_classes, input_shape = (32, 32, 3)):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape = input_shape, filters = 32, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv1')
        self.bnorm1 = tf.keras.layers.BatchNormalization(axis = 1, momentum = 0.9, epsilon = 1e-5, name = 'bnorm1')
        
        self.conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv2')
        self.bnorm2 = tf.keras.layers.BatchNormalization(axis = 1, momentum = 0.9, epsilon = 1e-5, name = 'bnorm2')
        
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size = [2,2], strides = [2,2], name = 'max_pool1')

        self.flatten1 = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc1')
        self.bnorm3 = tf.keras.layers.BatchNormalization(axis = 1, momentum = 0.9, epsilon = 1e-5, name = 'bnorm3')
        self.fc2 = tf.keras.layers.Dense(num_classes, name = 'fc2')

    def call(self, batch_input, is_training):
        x = self.conv1(batch_input)
        x = self.bnorm1(x, is_training)
        x = self.conv2(x)
        x = self.bnorm2(x, is_training)
        x = self.max_pool1(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.bnorm3(x, is_training)
        x = self.fc2(x)
        return x


# ### CNN7

# In[13]:


class CNN7(tf.keras.Model):
    def __init__(self, num_classes, input_shape = (32, 32, 3)):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(input_shape = input_shape, filters = 64, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv1')
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size = [2,2], strides = [2,2], name = 'max_pool1')
        self.conv2 = tf.keras.layers.Conv2D(filters = 128, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv2')
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size = [2,2], strides = [2,2], name = 'max_pool2')
        self.conv3 = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3],
                            padding = 'SAME', activation = tf.nn.relu, name = 'conv3')
        self.gpool1 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc1')
        self.bnorm1 = tf.keras.layers.BatchNormalization(axis = 1, momentum = 0.9,epsilon = 1e-5, name='bnorm1')
        self.fc2 = tf.keras.layers.Dense(num_classes, name = 'fc2')

    def call(self, batch_input, is_training):
        x = self.conv1(batch_input)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.gpool1(x)
        x = self.fc1(x)
        x = self.bnorm1(x, is_training)
        x = self.fc2(x)
        return x


# # Experiments

# ### Experiment #1 - CNN1

# In[16]:


trained_cnn1 = train(CNN1)


# In[18]:


test_acc1 = get_accuracy(trained_cnn1, X_test, y_test, X_test_iter, y_test_iter)
print ("CNN1 - Test Accuracy: {:.2f} %".format(test_acc1))


# ### Experiment #2 - CNN2

# In[ ]:


trained_cnn2 = train(CNN2)


# In[ ]:


test_acc2 = get_accuracy(trained_cnn2, X_test, y_test, X_test_iter, y_test_iter)
print ("CNN2 - Test Accuracy: {:.2f} %".format(test_acc2), end="\n\n")

trained_cnn2.summary()


# ### Experiment #3 - CNN3

# In[ ]:


cnn3_model = CNN3(NUM_CLASSES)
train(cnn3_model, num_epochs = 30)


# In[ ]:


test_acc3 = get_accuracy(cnn3_model, X_test, y_test, X_test_iter, y_test_iter)
print ("CNN3 - Test Accuracy: {:.2f} %".format(test_acc3), end="\n\n")

cnn3_model.summary()


# ### Experiment #4 - CNN4

# In[40]:


cnn4_model = CNN4(NUM_CLASSES)
train(cnn4_model, num_epochs = 30)


# In[41]:


test_acc4 = get_accuracy(cnn4_model, X_test, y_test, X_test_iter, y_test_iter)
print ("CNN4 - Test Accuracy: {:.2f} %".format(test_acc4), end="\n\n")

cnn4_model.summary()


# ### Experiment #5 - CNN5

# In[38]:


cnn5_model = CNN5(NUM_CLASSES)
train(cnn5_model, num_epochs = 30)


# In[39]:


test_acc5 = get_accuracy(cnn5_model, X_test, y_test, X_test_iter, y_test_iter)
print ("CNN5 - Test Accuracy: {:.2f} %".format(test_acc5), end="\n\n")

cnn5_model.summary()


# ### Experiment #6 - CNN6

# In[32]:


cnn6_model = CNN6(NUM_CLASSES)
train(cnn6_model, num_epochs = 30)


# In[34]:


test_acc6 = get_accuracy(cnn6_model, X_test, y_test, X_test_iter, y_test_iter)
print ("CNN6 - Test Accuracy: {:.2f} %".format(test_acc6), end="\n\n")

cnn6_model.summary()


# ### Experiment #7 - CNN7

# In[35]:


cnn7_model = CNN7(NUM_CLASSES)
train(cnn7_model, num_epochs = 30)


# In[36]:


test_acc7 = get_accuracy(cnn7_model, X_test, y_test, X_test_iter, y_test_iter)
print ("CNN7 - Test Accuracy: {:.2f} %".format(test_acc7), end="\n\n")

cnn7_model.summary()


# ## Max. Test Accuracy = 41.67 %
# CNN architecture - **CNN7** <br>
# Epochs: **7** <br>
# Iterations per epoch: **700** <br>
# Learning Rate: **3e-4**
