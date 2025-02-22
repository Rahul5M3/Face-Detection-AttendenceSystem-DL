from tensorflow.keras.models import Model, load_model   
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
from keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from scipy.spatial.distance import cdist
from datetime import datetime

def mongoDB_connection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["employee_attendence"]

    print("MongoDB connected successfully!")
    return db

@register_keras_serializable()
class FaceDetector(Model):
    def __init__(self, faceDetector, **kwargs):
        super().__init__(**kwargs)
        self.model=faceDetector

    def compile(self, opt, classLoss, regressLoss,  **kwargs):
            # super().compile(**kwargs)
            # self.opt=opt
            # self.cLoss=classLoss
            # self.lLoss=regressLoss

            super().compile(
                  optimizer=opt,
                  loss=[classLoss,regressLoss],
                  **kwargs
            )
            self.opt=opt
            self.cLoss=classLoss
            self.lLoss=regressLoss

            # self.cLoss=loss.get('classification')
            # self.lLoss=loss.get('regression')
            # super().compile(optimizer=opt, loss={'classification': self.cLoss, 'regression': self.lLoss},**kwargs)
            # self.opt=opt

    def train_step(self, batch, **kwargs):
        X,y=batch
        # Ensure the shape of X is (batch_size, 120, 120, 3)
        X = tf.ensure_shape(X, [None, 120, 120, 3])

        y = list(y)

        # # Ensure y has the expected shape (batch_size, num_classes) for class labels
        y[0] = tf.ensure_shape(y[0], [None, 1])  # Adjust num_classes as needed
        y[1] = tf.ensure_shape(y[1], [None, 4])  # Adjust num_boxes as needed

        y = tuple(y)


        # print(f"Rank of X: {tf.rank(X)}")
        # print(f"Rank of y: {tf.rank(y)}")
        with tf.GradientTape()  as tape:
              classes,coordinates,features= self.model(X,training=True)
              classLoss=self.cLoss(y[0],classes)
              localizationLoss=self.lLoss(tf.cast(y[1],tf.float32),coordinates)
              totalLoss=0.5*classLoss+localizationLoss
              grad=tape.gradient(totalLoss,self.model.trainable_variables)

        opt.apply_gradients(zip(grad,self.model.trainable_variables))   
        return {'totalLoss':totalLoss, 'classLoss':classLoss, 'regressLoss':localizationLoss }

    def test_step(self, batch, **knwargs):
              
            X,y=batch

            # Ensure the shape of X is (batch_size, 120, 120, 3)
            X = tf.ensure_shape(X, [None, 120, 120, 3])

            y = list(y)

            # # Ensure y has the expected shape (batch_size, num_classes) for class labels
            y[0] = tf.ensure_shape(y[0], [None, 1])  # Adjust num_classes as needed
            y[1] = tf.ensure_shape(y[1], [None, 4])  # Adjust num_boxes as needed

            y = tuple(y)

            classes,coordinates= self.model(X,training=True)
            classLoss=self.cLoss(y[0],classes)
            localizationLoss=self.lLoss(tf.cast(y[1],tf.float32),coordinates)
            totalLoss=0.5*classLoss+localizationLoss

            return {'totalLoss':totalLoss, 'classLoss':classLoss, 'regressLoss':localizationLoss}
    
    def call(self, X, **kwargs):
          return self.model(X, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({"faceDetector": self.model.to_json()})  
        return config

    @classmethod
    def from_config(cls, config):
        from tensorflow.keras.models import model_from_json
        faceDetector = config.pop("faceDetector") 
        faceDetector = model_from_json(faceDetector)
        return cls(faceDetector, **config)
            

def empProcess(model):
    cap=cv2.VideoCapture(0)


    # emp_name=input("Name of Employee")
    # emp_id=input("Id of Employee")

    frames4=[]

    frame_count = 0

    while cap.isOpened():
        _,frame=cap.read()
        frame = frame[50:500, 50:500,:]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120,120))
        yPred=model.predict(np.expand_dims(resized/255,0))
        coords=yPred[1][0]

        frame_count=frame_count+1

        x,y,h,w=np.multiply(coords, [450,450,450,450]).astype(int)
        frameh=frame[y:y+h, x:x+w].copy()
        
        if yPred[0]>0.7:
            cv2.rectangle(frame, tuple(np.multiply(coords[:2], [450,450]).astype(int)), tuple(np.multiply(coords[2:], [450,450]).astype(int)), (255,200,0), 2)

            cv2.rectangle(frame, 
                        tuple(np.add(np.multiply(coords[:2], [450,450]).astype(int), 
                                        [0,-30])),
                        tuple(np.add(np.multiply(coords[:2], [450,450]).astype(int),
                                        [80,0])), 
                                (255,0,0), -1)
            
            cv2.putText(frame, 'face', tuple(np.add(np.multiply(coords[:2], [450,450]).astype(int),
                                                [0,-5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('Face Track', frame)

        if frame_count%8==0 and len(frames4)<4 : 
            frames4.append(frameh)

        if (cv2.waitKey(1) & 0xFF == ord('b')) or len(frames4)>=4:
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames4


def load_facenet():
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import GlobalAveragePooling2D
    # model_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
    # efficientnet = hub.KerasLayer(model_url, trainable=False)

    base_model = EfficientNetB0(weights='imagenet', include_top=False)
    base_model.trainable=False

    output_layer = GlobalAveragePooling2D()(base_model.output)

    dense_layer=Dense(units=128, activation=None, name="embedding")(output_layer)
    
    model = Model(inputs=base_model.input, outputs=dense_layer)
    return model


def preprocess_face(face):
    face=cv2.resize(face,(224,224))
    face=face.astype('float32')/255.0
    face=np.expand_dims(face,axis=0)
    return face

def extract_embedding(model,face):
    face=preprocess_face(face)
    embedding=model.predict(face)
    return embedding

def store_embeddings(efficientnetModel, frames4):
    framesEmbeddings=[]
    for face in frames4:
        face=extract_embedding(efficientnetModel, face)
        framesEmbeddings.append(face)
    return framesEmbeddings    

def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def main():
    db=mongoDB_connection()

    action=input('s for storing and f for finding')

    current_year = str(datetime.now().year)
    months = {
        "January": 31, "February": 28, "March": 31, "April": 30,
        "May": 31, "June": 30, "July": 31, "August": 31,
         "September": 30, "October": 31, "November": 30, "December": 31
    }

    if int(current_year) % 4 == 0 and (int(current_year) % 100 != 0 or int(current_year) % 400 == 0):
        months["February"] = 29

    if action=='s':
        emp_name=input('Employee Name')
        emp_id=input('Employee Id')

        model = load_model("../Model/modelFaceDetection.keras", custom_objects={"FaceDetector": FaceDetector}, compile=False)
        cropFrames=empProcess(model)
        efficientnetModel=load_facenet()
        frameEmbeddings=store_embeddings(efficientnetModel, cropFrames)

        frameEmbeddings = [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in frameEmbeddings]

        res=db.employee.insert_one({"name":emp_name, "_id":emp_id, "embeddings":frameEmbeddings})
        # res=db.empAttendence.insert_one({"name":emp_name, "_id":emp_id,})

        new_doc = {
            "_id": res.inserted_id,
            current_year: {month: [[] for _ in range(days)] for month, days in months.items()}
        }
        db.empAttendence.insert_one(new_doc)

        if res.inserted_id:
            print("Sucessfully Inserted ID : ", res.inserted_id)
        else :
            print("Employee insertion failed")

    elif action=='f':
        model = load_model("../Model/modelFaceDetection.keras", custom_objects={"FaceDetector": FaceDetector}, compile=False)
        cropFrames=empProcess(model)
        efficientnetModel=load_facenet()
        frameEmbeddings=store_embeddings(efficientnetModel, cropFrames)  

        frameEmbeddings = [np.array(embedding, dtype=np.float32) if isinstance(embedding, list) else embedding for embedding in frameEmbeddings]
        frameEmbeddings = np.array(frameEmbeddings, dtype=np.float32)   
        frameEmbeddings = frameEmbeddings.squeeze(1)
        frameEmbeddings=normalize_embeddings(frameEmbeddings)

        bestScore=float('-inf')
        bestMatch=None

        for document in db.employee.find({},{'embeddings':1, '_id':1}):
            embedding = [np.array(embedding, dtype=np.float32) if isinstance(embedding, list) else embedding for embedding in document["embeddings"]]
            embedding=np.array(embedding, dtype=np.float32)
            embedding = embedding.squeeze(1)
            embedding=normalize_embeddings(embedding)

            similarity_matrix = 1 - cdist(frameEmbeddings, embedding, metric='cosine')

            max_sim = np.max(similarity_matrix)

            if max_sim > bestScore:  
                bestScore = max_sim
                bestMatch = document["_id"]

        print(f"Best match: {bestMatch} with similarity score: {bestScore}")   

        # attendence part

        # current_year = str(datetime.now().year)
        # months = {
        #     "January": 31, "February": 28, "March": 31, "April": 30,
        #     "May": 31, "June": 30, "July": 31, "August": 31,
        #     "September": 30, "October": 31, "November": 30, "December": 31
        # }

        # if int(current_year) % 4 == 0 and (int(current_year) % 100 != 0 or int(current_year) % 400 == 0):
        #     months["February"] = 29

        doc = db.empAttendence.find_one({"_id": bestMatch})
        
        if doc:
            update_fields = {} 

            if current_year not in doc:
                update_fields[current_year] = {month: [[] for _ in range(days)] for month, days in months.items()}  
            else :
                for month, days in months.items():
                    if month not in doc[current_year]:
                        update_fields[f"{current_year}.{month}"] = [[] for _ in range(days)]   

            if update_fields:
                db.empAttendence.update_one(
                    {"_id": "some_unique_id"},
                    {"$set": update_fields}
                ) 
        else :
            new_doc = {
                "_id": bestMatch,
                current_year: {month: [[] for _ in range(days)] for month, days in months.items()}
            }
            db.empAttendence.insert_one(new_doc)       

        current_month = datetime.now().strftime("%B")
        current_day = datetime.now().day - 1

        db.empAttendence.update_one(
            {"_id": bestMatch},
            {"$push": {f"{current_year}.{current_month}.{current_day}": {'status':'present'}}}
        )                 

    
         

if __name__ == "__main__":
    main()