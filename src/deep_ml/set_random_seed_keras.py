
import os
import random
import numpy as np
import tensorflow as tf 
import keras.backend as K


# seed_value= 0

# # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
# import os
# os.environ['PYTHONHASHSEED']=str(seed_value)

# # 2. Set `python` built-in pseudo-random generator at a fixed value
# import random
# random.seed(seed_value)

# # 3. Set `numpy` pseudo-random generator at a fixed value
# import numpy as np
# np.random.seed(seed_value)

# # 4. Set `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
# tf.set_random_seed(seed_value)

# # 5. Configure a new global `tensorflow` session
# from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, 
#                               inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)


def set_seed_keras(seed=1234):
	random.seed(seed)
	os.environ['PYTHONHASHSEED']=str(seed)
	np.random.seed(seed)
	tf.set_random_seed(seed)
	session_conf = tf.ConfigProto(
		intra_op_parallelism_threads=1, 
	    inter_op_parallelism_threads=1)
	sess = tf.Session(
		graph=tf.get_default_graph(), 
		config=session_conf)
	K.set_session(sess)



def set_seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
# set_seed_torch()
