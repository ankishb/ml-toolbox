

from sklearn import preprocessing
X_train_ordinal = X_train.values
X_test_ordinal = X_test.values
for i in range(X_train_ordinal.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(data.iloc[:,features].iloc[:, i])
    les.append(le)
    X_train_ordinal[:, i] = le.transform(X_train_ordinal[:, i])
    X_test_ordinal[:, i] = le.transform(X_test_ordinal[:, i])




from sklearn.feature_extraction import FeatureHasher
X_train_hash = copy.copy(X_train)
X_test_hash = copy.copy(X_test)
for i in range(X_train_hash.shape[1]):
    X_train_hash.iloc[:,i]=X_train_hash.iloc[:,i].astype('str')
for i in range(X_test_hash.shape[1]):
    X_test_hash.iloc[:,i]=X_test_hash.iloc[:,i].astype('str')
h = FeatureHasher(n_features=100,input_type="string")
X_train_hash = h.transform(X_train_hash.values)
X_test_hash = h.transform(X_test_hash.values)





import copy
X_train_rare = copy.copy(X_train)
X_test_rare = copy.copy(X_test)
X_train_rare["test"]=0
X_test_rare["test"]=1
temp_df = pandas.concat([X_train_rare,X_test_rare],axis=0)
names = list(X_train_rare.columns.values)
temp_df = pandas.concat([X_train_rare,X_test_rare],axis=0)
for i in names:
    temp_df.loc[temp_df[i].value_counts()[temp_df[i]].values < 20, i] = "RARE_VALUE"
for i in range(temp_df.shape[1]):
    temp_df.iloc[:,i]=temp_df.iloc[:,i].astype('str')
X_train_rare = temp_df[temp_df["test"]=="0"].iloc[:,:-1].values
X_test_rare = temp_df[temp_df["test"]=="1"].iloc[:,:-1].values
for i in range(X_train_rare.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(temp_df.iloc[:,:-1].iloc[:, i])
    les.append(le)
    X_train_rare[:, i] = le.transform(X_train_rare[:, i])
    X_test_rare[:, i] = le.transform(X_test_rare[:, i])
enc.fit(X_train_rare)
X_train_rare = enc.transform(X_train_rare)
X_test_rare = enc.transform(X_test_rare)
l.fit(X_train_rare,y_train)
y_pred = l.predict_proba(X_test_rare)
print(log_loss(y_test,y_pred))
r.fit(X_train_rare,y_train)
y_pred = r.predict_proba(X_test_rare)
print(log_loss(y_test,y_pred))
print(X_train_rare.shape)





X_train_count = copy.copy(X_train)
X_test_count = copy.copy(X_test)
X_train_count["test"]=0
X_test_count["test"]=1
temp_df = pandas.concat([X_train_count,X_test_count],axis=0)
for i in range(temp_df.shape[1]):
    temp_df.iloc[:,i]=temp_df.iloc[:,i].astype('category')
X_train_count=temp_df[temp_df["test"]==0].iloc[:,:-1]
X_test_count=temp_df[temp_df["test"]==1].iloc[:,:-1]
for i in range(X_train_count.shape[1]):
    counts = X_train_count.iloc[:,i].value_counts()
    counts = counts.sort_index()
    counts = counts.fillna(0)
    counts += np.random.rand(len(counts))/1000
    X_train_count.iloc[:,i].cat.categories = counts
    X_test_count.iloc[:,i].cat.categories = counts

	
	
	
	
	

	
	

X_train_ctr = copy.copy(X_train)
X_test_ctr = copy.copy(X_test)
X_train_ctr["test"]=0
X_test_ctr["test"]=1
temp_df = pandas.concat([X_train_ctr,X_test_ctr],axis=0)
for i in range(temp_df.shape[1]):
    temp_df.iloc[:,i]=temp_df.iloc[:,i].astype('category')
X_train_ctr=temp_df[temp_df["test"]==0].iloc[:,:-1]
X_test_ctr=temp_df[temp_df["test"]==1].iloc[:,:-1]
temp_df = pandas.concat([X_train_ctr,y_train],axis=1)
names = list(X_train_ctr.columns.values)
for i in names:
    means = temp_df.groupby(i)['click'].mean()
    means = means.fillna(sum(temp_df['click'])/len(temp_df['click']))
    means += np.random.rand(len(means))/1000
    X_train_ctr[i].cat.categories = means
    X_test_ctr[i].cat.categories = means


	
	
	
	
	
	
from gensim.models.word2vec import Word2Vec
from random import shuffle
size=6
window=8
x_w2v = copy.deepcopy(data.iloc[:,features])
names = list(x_w2v.columns.values)
for i in names:
    x_w2v[i]=x_w2v[i].astype('category')
    x_w2v[i].cat.categories = ["Feature %s %s" % (i,g) for g in x_w2v[i].cat.categories]
x_w2v = x_w2v.values.tolist()
for i in x_w2v:
    shuffle(i)
w2v = Word2Vec(x_w2v,size=size,window=window)

X_train_w2v = copy.copy(X_train)
X_test_w2v = copy.copy(X_test)
for i in names:
    X_train_w2v[i]=X_train_w2v[i].astype('category')
    X_train_w2v[i].cat.categories = ["Feature %s %s" % (i,g) for g in X_train_w2v[i].cat.categories]
for i in names:
    X_test_w2v[i]=X_test_w2v[i].astype('category')
    X_test_w2v[i].cat.categories = ["Feature %s %s" % (i,g) for g in X_test_w2v[i].cat.categories]
X_train_w2v = X_train_w2v.values
X_test_w2v = X_test_w2v.values
x_w2v_train = np.random.random((len(X_train_w2v),size*X_train_w2v.shape[1]))
for j in range(X_train_w2v.shape[1]):
    for i in range(X_train_w2v.shape[0]):
        if X_train_w2v[i,j] in w2v:
            x_w2v_train[i,j*size:(j+1)*size] = w2v[X_train_w2v[i,j]]

x_w2v_test = np.random.random((len(X_test_w2v),size*X_test_w2v.shape[1]))
for j in range(X_test_w2v.shape[1]):
    for i in range(X_test_w2v.shape[0]):
        if X_test_w2v[i,j] in w2v:
            x_w2v_test[i,j*size:(j+1)*size] = w2v[X_test_w2v[i,j]]


			
			
			
			
			
			
			
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape
from keras.layers import Merge
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
import h5py

X_train_dnn = copy.copy(X_train).values
X_test_dnn = copy.copy(X_test).values
    
les = []
for i in range(X_train_dnn.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(data.iloc[:,features].iloc[:, i])
    les.append(le)
    X_train_dnn[:, i] = le.transform(X_train_dnn[:, i])
    X_test_dnn[:, i] = le.transform(X_test_dnn[:, i])
    
def split_features(X):
    X_list = []

	C1 = X[..., [0]]
		X_list.append(C1)

	banner_pos = X[..., [1]]
		X_list.append(banner_pos)

	site_id = X[..., [2]]
		X_list.append(site_id)
		
		site_domain = X[..., [3]]
		X_list.append(site_domain)

	site_category = X[..., [4]]
		X_list.append(site_category)

	app_id = X[..., [5]]
		X_list.append(app_id)

	app_domain = X[..., [6]]
		X_list.append(app_domain)

	app_category = X[..., [7]]
		X_list.append(app_category)
		
		device_id = X[..., [8]]
		X_list.append(device_id)

	device_model = X[..., [9]]
		X_list.append(device_model)

	device_type = X[..., [10]]
		X_list.append(device_type)
		
		device_conn_type = X[..., [11]]
		X_list.append(device_conn_type)

	C14 = X[..., [12]]
		X_list.append(C14)
		
		C15 = X[..., [13]]
		X_list.append(C15)

	C16 = X[..., [14]]
		X_list.append(C16)

	C17 = X[..., [15]]
		X_list.append(C17)

	C18 = X[..., [16]]
		X_list.append(C18)

	C19 = X[..., [17]]
		X_list.append(C19)
		
		C20 = X[..., [18]]
		X_list.append(C20)

	C21 = X[..., [19]]
		X_list.append(C21)

	return X_list

class NN_with_EntityEmbedding(object):

def __init__(self, X_train, y_train, X_val, y_val):
        self.nb_epoch = 10
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

def __build_keras_model(self):
        models = []

model_C1= Sequential()
        model_C1.add(Embedding(len(les[0].classes_), 3, input_length=1))
        model_C1.add(Reshape(target_shape=(3,)))
        models.append(model_C1)

model_banner_pos = Sequential()
        model_banner_pos.add(Embedding(len(les[1].classes_), 3, input_length=1))
        model_banner_pos.add(Reshape(target_shape=(3,)))
        models.append(model_banner_pos)
        
        model_site_id = Sequential()
        model_site_id.add(Embedding(len(les[2].classes_), 8, input_length=1))
        model_site_id.add(Reshape(target_shape=(8,)))
        models.append(model_site_id)
        
        site_domain = Sequential()
        site_domain.add(Embedding(len(les[3].classes_), 8, input_length=1))
        site_domain.add(Reshape(target_shape=(8,)))
        models.append(site_domain)

site_category = Sequential()
        site_category.add(Embedding(len(les[4].classes_), 3, input_length=1))
        site_category.add(Reshape(target_shape=(3,)))
        models.append(site_category)

app_id = Sequential()
        app_id.add(Embedding(len(les[5].classes_), 8, input_length=1))
        app_id.add(Reshape(target_shape=(8,)))
        models.append(app_id)

app_domain = Sequential()
        app_domain.add(Embedding(len(les[6].classes_), 4, input_length=1))
        app_domain.add(Reshape(target_shape=(4,)))
        models.append(app_domain)
        
        app_category = Sequential()
        app_category.add(Embedding(len(les[7].classes_), 3, input_length=1))
        app_category.add(Reshape(target_shape=(3,)))
        models.append(app_category)
        
        device_id = Sequential()
        device_id.add(Embedding(len(les[8].classes_), 10, input_length=1))
        device_id.add(Reshape(target_shape=(10,)))
        models.append(device_id)
        
        device_model = Sequential()
        device_model.add(Embedding(len(les[9].classes_), 8, input_length=1))
        device_model.add(Reshape(target_shape=(8,)))
        models.append(device_model)
        
        device_type = Sequential()
        device_type.add(Embedding(len(les[10].classes_), 2, input_length=1))
        device_type.add(Reshape(target_shape=(2,)))
        models.append(device_type)
        
        device_conn_type = Sequential()
        device_conn_type.add(Embedding(len(les[11].classes_), 2, input_length=1))
        device_conn_type.add(Reshape(target_shape=(2,)))
        models.append(device_conn_type)

C14 = Sequential()
        C14.add(Embedding(len(les[12].classes_), 8, input_length=1))
        C14.add(Reshape(target_shape=(8,)))
        models.append(C14)
        
        C15 = Sequential()
        C15.add(Embedding(len(les[13].classes_), 3, input_length=1))
        C15.add(Reshape(target_shape=(3,)))
        models.append(C15)
        
        C16 = Sequential()
        C16.add(Embedding(len(les[14].classes_), 3, input_length=1))
        C16.add(Reshape(target_shape=(3,)))
        models.append(C16)
        
        C17 = Sequential()
        C17.add(Embedding(len(les[15].classes_), 4, input_length=1))
        C17.add(Reshape(target_shape=(4,)))
        models.append(C17)
        
        C18 = Sequential()
        C18.add(Embedding(len(les[16].classes_), 2, input_length=1))
        C18.add(Reshape(target_shape=(2,)))
        models.append(C18)
        
        C19 = Sequential()
        C19.add(Embedding(len(les[17].classes_), 4, input_length=1))
        C19.add(Reshape(target_shape=(4,)))
        models.append(C19)
        
        C20 = Sequential()
        C20.add(Embedding(len(les[18].classes_), 5, input_length=1))
        C20.add(Reshape(target_shape=(5,)))
        models.append(C20)
        
        C21 = Sequential()
        C21.add(Embedding(len(les[19].classes_), 4, input_length=1))
        C21.add(Reshape(target_shape=(4,)))
        models.append(C21)

self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dense(150, kernel_initializer='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(250, kernel_initializer='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

self.model.compile(loss='binary_crossentropy',
              optimizer='adam',
             metrics=['acc'])
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.preprocessing(X_train), y_train,
                       validation_data=(self.preprocessing(X_val), y_val),
                       epochs=self.nb_epoch, batch_size=128,
                       )

dnn = NN_with_EntityEmbedding(X_train_dnn, y_train, X_test_dnn, y_test)   
weights = dnn.model.get_weights()

n = 0
for i in range(0,40,2):
    n+=(weights[i][0].shape)[1]
    
x_dnn_train = np.random.random((len(X_train_dnn),n))
start_ind=0
for j in range(X_train_dnn.shape[1]):
    mat = weights[j*2][0]
    dim = mat.shape[1]
    for i in range(X_train_dnn.shape[0]):
        x_dnn_train[i,start_ind:start_ind+dim]=mat[X_train_dnn[i,j]]
    start_ind += dim

x_dnn_test = np.random.random((len(X_test_dnn),n))
start_ind=0
for j in range(X_test_dnn.shape[1]):
    mat = weights[j*2][0]
    dim = mat.shape[1]
    for i in range(x_dnn_test.shape[0]):
        x_dnn_test[i,start_ind:start_ind+dim]=mat[X_test_dnn[i,j]]
    start_ind += dim

l.fit(x_dnn_train,y_train)
y_pred = l.predict_proba(x_dnn_test)
print(log_loss(y_test,y_pred))

r.fit(x_dnn_train,y_train)
y_pred = r.predict_proba(x_dnn_test)
print(log_loss(y_test,y_pred))



